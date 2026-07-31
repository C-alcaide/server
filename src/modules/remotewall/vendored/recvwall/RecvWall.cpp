// recv_wall.lib implementation. Decode-side mirror of cloudXR's nvenc_tile.lib.
//
// Structure factored from cloudXR's live receivers (decklink_player.cpp /
// screen_player.cpp): one rx thread drains the UDP socket into the RtpFec
// reassembler; each reassembled access unit is queued per tile (bounded,
// drop-oldest to stay at the live edge); one NVDEC worker per tile decodes and
// inserts into the aligner keyed by the in-band SyncMeta globalFrameIndex; the
// worker that lands the LAST tile of a wall frame composites it (outside the
// aligner lock) and publishes it as the latest-wins snapshot clients poll.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <cuda.h>
#include <cuda_runtime.h>   // cudaGetLastError: detect failed convert-kernel launches
#include <d3d11.h>
#include <cudaD3D11.h>      // CUDA<->D3D11 interop for the zero-copy texture output
#include "NvDecoder/NvDecoder.h"
#include "ColorSpace.h" // BGRA32/RGBA32 + Nv12ToColor32 (kernel in ColorSpace.obj)
#include "sync/SyncMeta.h"
#include "net/RtpFec.h"
#include "RecvWall.h"
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#pragma comment(lib, "ws2_32.lib")

// NvDecoder.cpp expects this global (each cloudXR tool defines its own copy).
simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger();

// Instantiated in ColorSpace.cu (compiled with nvcc, linked into the lib).
template <class COLOR32> void Nv12ToColor32(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpBgra,
    int nBgraPitch, int nWidth, int nHeight, int iMatrix, bool video_full_range);
// 10/12-bit P016 -> 16-bit 4ch (RGBA64/BGRA64), also instantiated in ColorSpace.cu.
template <class COLOR64> void P016ToColor64(uint8_t* dpP016, int nP016Pitch, uint8_t* dpBgra,
    int nBgraPitch, int nWidth, int nHeight, int iMatrix, bool video_full_range);

// cfg.codec: 0 = HEVC (default), 1 = H.264, 2 = AV1.
static cudaVideoCodec pickCudaCodec(int c)
{
	return c == 1 ? cudaVideoCodec_H264 : c == 2 ? cudaVideoCodec_AV1 : cudaVideoCodec_HEVC;
}
// Extract every embedded SyncMeta from one access unit, picking the codec-specific
// carriage (Annex-B SEI for H.264/HEVC, metadata OBU for AV1).
template <class Cb>
static void scanMeta(int codec, const uint8_t* d, size_t n, Cb cb)
{
	if (codec == 2)
		sync::scanAv1Obu(d, n, cb);
	else
		sync::scanAnnexB(d, n, codec == 0, cb);
}
// Map the NVDEC surface bit depth to the composite's bytes/pixel: 8-bit -> RGBA8
// (4 B/px), 10/12-bit -> RGBA16 (8 B/px).
static int compBppFor(int bitDepth) { return bitDepth > 8 ? 8 : 4; }

// Pick the YUV->RGB matrix for the P016 (HDR) convert from the SEI colour tag.
// HDR walls are BT.2020; SDR-tagged 10-bit falls back to BT.709.
static int matrixFor(const char* cs)
{
	std::string s = cs ? cs : "";
	for (auto& ch : s) ch = (char)std::toupper((unsigned char)ch);
	if (s.find("2020") != std::string::npos) return ColorSpaceStandard_BT2020;
	if (s.find("601") != std::string::npos) return ColorSpaceStandard_BT601;
	return ColorSpaceStandard_BT709;
}


namespace {

// NV12 -> 8-bit 4ch at (ox,oy) in a top-down wall buffer. roiW/roiH clamp the
// converted region to the tile cell (the decoder surface can be larger than the
// tile). yL/cL are range LUTs (identity when full-range). matrix selects the
// YCbCr->RGB coefficients (BT.601/709/2020) from the stream's colour tag.
static void nv12ToWall(const uint8_t* nv12, int srcW, int srcH, int roiW, int roiH,
                       uint8_t* dst, int dstStride, int ox, int oy, int dstW, int dstH,
                       const uint8_t* yL, const uint8_t* cL, bool rgba, int matrix)
{
	const uint8_t* Y = nv12;
	const uint8_t* UV = nv12 + (size_t)srcW * srcH;
	if (roiW > srcW) roiW = srcW;
	if (roiH > srcH) roiH = srcH;
	const int bIdx = rgba ? 2 : 0, rIdx = rgba ? 0 : 2;
	// Full-range YCbCr->RGB, <<16 fixed point: {r<-v, g<-u, g<-v, b<-u}.
	int cvr, cug, cvg, cub;
	if (matrix == ColorSpaceStandard_BT2020)      { cvr = 96636;  cug = 10784; cvg = 37443; cub = 123304; }
	else if (matrix == ColorSpaceStandard_BT709)  { cvr = 103211; cug = 12276; cvg = 30678; cub = 121610; }
	else                                          { cvr = 91881;  cug = 22554; cvg = 46802; cub = 116130; } // BT.601
	for (int y = 0; y < roiH; ++y) {
		int dy = oy + y; if (dy < 0 || dy >= dstH) continue;
		const uint8_t* yr = Y + (size_t)y * srcW;
		const uint8_t* uvr = UV + (size_t)(y / 2) * srcW;
		uint8_t* drow = dst + (size_t)dy * dstStride + (size_t)ox * 4;
		for (int x = 0; x < roiW; ++x) {
			if (ox + x < 0 || ox + x >= dstW) { drow += 4; continue; }
			int yy = yL[yr[x]];
			int uv = (x & ~1);
			int u = cL[uvr[uv]] - 128, v = cL[uvr[uv + 1]] - 128;
			int r = yy + ((cvr * v) >> 16);
			int g = yy - ((cug * u + cvg * v) >> 16);
			int b = yy + ((cub * u) >> 16);
			drow[bIdx] = (uint8_t)(b < 0 ? 0 : b > 255 ? 255 : b);
			drow[1]    = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : g);
			drow[rIdx] = (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
			drow[3] = 255; drow += 4;
		}
	}
}

// ---------------------------------------------------------------------------
// Cross-instance sync board: named shared memory + named mutex, so instances
// in DIFFERENT PROCESSES (Assimilate hosts plugin instances in separate
// processes) can frame-lock their presentation. Self-contained Win32 on
// purpose - recv_wall.lib keeps zero extra dependencies.
// ---------------------------------------------------------------------------
static const uint64_t SEI_NONE = ~0ull;

#pragma pack(push, 4)
struct SyncSlot {                     // 64 bytes
	uint32_t active;                  // claimed (mutated only under the named mutex)
	uint32_t pid;
	uint32_t instanceNonce;           // pid-reuse guard
	uint32_t epochResets;             // diagnostic
	uint64_t lastCompleteSei;         // SEI_NONE => JOINING (does not gate)
	uint64_t completedBits;           // bit k = completed (lastCompleteSei - k)
	uint64_t heartbeatTickMs;         // process/rx alive (slot reclaim only)
	uint64_t progressTickMs;          // last wall completion (syncTimeoutMs gates on this)
	uint64_t reserved[3];
};
struct SyncBoard {
	uint32_t magic;                   // 'RWSB'
	uint32_t version;                 // 1
	uint32_t slotCount;               // kSyncSlots
	uint32_t _pad;
	SyncSlot slots[16];
};
#pragma pack(pop)
static const uint32_t kSyncMagic = 0x42535752; // 'RWSB'
static const int kSyncSlots = 16;

struct SyncBoardConn {
	HANDLE hMap = nullptr, hMtx = nullptr;
	SyncBoard* bd = nullptr;
	int mySlot = -1;
	uint32_t nonce = 0;

	bool lock() {
		DWORD r = WaitForSingleObject(hMtx, 2000);
		return r == WAIT_OBJECT_0 || r == WAIT_ABANDONED;  // abandoned: scalars stay sane
	}
	void unlock() { ReleaseMutex(hMtx); }

	static bool pidDead(uint32_t pid) {
		HANDLE p = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, pid);
		if (!p) return GetLastError() == ERROR_INVALID_PARAMETER;   // no such pid
		bool dead = WaitForSingleObject(p, 0) == WAIT_OBJECT_0;
		CloseHandle(p);
		return dead;
	}

	bool open(const char* group) {
		char name[96], safe[32];
		int n = 0;
		for (const char* p = group; *p && n < 31; ++p) {
			char c = *p;
			safe[n++] = (isalnum((unsigned char)c) || c == '-' || c == '_') ? c : '_';
		}
		safe[n] = 0;
		std::snprintf(name, sizeof name, "Local\\RemoteWallSyncMtx_%s", safe);
		hMtx = CreateMutexA(nullptr, FALSE, name);
		if (!hMtx) return false;
		std::snprintf(name, sizeof name, "Local\\RemoteWallSync_%s", safe);
		hMap = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
		                          0, sizeof(SyncBoard), name);
		if (!hMap) { CloseHandle(hMtx); hMtx = nullptr; return false; }
		bd = (SyncBoard*)MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SyncBoard));
		if (!bd) { close(); return false; }
		if (!lock()) { close(); return false; }
		if (bd->magic == 0) {          // fresh mapping is zero-filled: initialize
			bd->magic = kSyncMagic; bd->version = 1; bd->slotCount = kSyncSlots;
		}
		if (bd->magic != kSyncMagic || bd->version != 1) { unlock(); close(); return false; }
		const uint64_t now = GetTickCount64();
		for (int i = 0; i < kSyncSlots; ++i) {      // reclaim dead members
			SyncSlot& s = bd->slots[i];
			if (s.active && (pidDead(s.pid) || now - s.heartbeatTickMs > 30000))
				std::memset(&s, 0, sizeof(s));
		}
		for (int i = 0; i < kSyncSlots; ++i) {
			SyncSlot& s = bd->slots[i];
			if (!s.active) {
				std::memset(&s, 0, sizeof(s));
				s.active = 1; s.pid = GetCurrentProcessId();
				LARGE_INTEGER qpc; QueryPerformanceCounter(&qpc);
				nonce = s.instanceNonce = (uint32_t)(qpc.LowPart ^ (s.pid << 16));
				s.lastCompleteSei = SEI_NONE;       // JOINING
				s.heartbeatTickMs = s.progressTickMs = now;
				mySlot = i;
				break;
			}
		}
		unlock();
		if (mySlot < 0) { close(); return false; }  // board full of live members
		return true;
	}

	void close() {
		if (bd && mySlot >= 0 && lock()) {
			if (bd->slots[mySlot].instanceNonce == nonce)
				std::memset(&bd->slots[mySlot], 0, sizeof(SyncSlot));
			unlock();
		}
		if (bd) UnmapViewOfFile(bd);
		if (hMap) CloseHandle(hMap);
		if (hMtx) CloseHandle(hMtx);
		bd = nullptr; hMap = hMtx = nullptr; mySlot = -1;
	}
};

static void fillInfo(RecvWallFrameInfo& out, const sync::SyncMeta& m, uint64_t sendTimeNs,
                     int wallW, int wallH)
{
	std::memset(&out, 0, sizeof(out));
	out.globalFrameIndex = m.globalFrameIndex;
	out.createTimeNsUtc = m.createTimeNsUtc;
	out.sendTimeNs = sendTimeNs;
	out.wallW = (unsigned)wallW; out.wallH = (unsigned)wallH;
	out.tileW = m.tileW; out.tileH = m.tileH;
	out.gridCols = m.gridCols; out.gridRows = m.gridRows;
	out.fpsNum = m.fpsNum; out.fpsDen = m.fpsDen;
	std::memcpy(out.colorSpace, m.colorSpace, 16);
	out.tcHours = m.tcHours; out.tcMinutes = m.tcMinutes; out.tcSeconds = m.tcSeconds;
	out.tcFrames = m.tcFrames; out.tcDrop = m.tcDrop; out.tcValid = m.tcValid;
	static_assert(sizeof(sync::CameraMeta) == 62 * sizeof(float), "camera layout");
	std::memcpy(out.cam, &m.cam, sizeof(out.cam));
}

} // namespace

struct RecvWallHandle
{
	RecvWallConfig cfg{};
	bool wsa = false;
	SOCKET sock = INVALID_SOCKET;
	CUdevice cuDev = 0;
	CUcontext ctx = nullptr;

	std::atomic<bool> running{true};
	std::thread rx;
	rtpfec::Reassembler* reasm = nullptr;
	std::mutex reasmMtx;                   // guards reasm swap (sender restart) vs stats read
	rtpfec::Reassembler::Callback auCb;   // kept so rx can rebuild reasm on sender restart
	uint32_t rxHwm = 0;                   // transport frame high-water mark (rx thread only)

	struct AuJob { uint32_t frame; uint64_t sendTimeNs; std::vector<uint8_t> au; };
	struct TileQ { std::mutex m; std::condition_variable cv; std::deque<AuJob> q; };
	std::map<uint16_t, TileQ*> tileQs; std::mutex tqMtx;
	std::vector<std::thread> workers;

	// Geometry + aligner (guarded by pendingMtx, like decklink_player).
	std::mutex pendingMtx;
	int cols = 0, rows = 0, tileW = 0, tileH = 0, wallW = 0, wallH = 0;
	unsigned fpsNum = 0, fpsDen = 1;
	int expectedTiles = 0;
	std::atomic<bool> haveGeo{false};
	std::atomic<long long> lastComplete{-1};
	// CPU path: frame -> tile -> NV12 (+ the frame's meta/sendTime from any tile)
	std::map<uint64_t, std::map<uint16_t, std::vector<uint8_t>>> pending;
	std::map<uint64_t, std::pair<sync::SyncMeta, uint64_t>> pendingInfo;
	// GPU path: frame -> (device composite, tiles written, meta, sendTime)
	struct GFrame { CUdeviceptr dComp = 0; int written = 0; sync::SyncMeta meta; uint64_t sendTimeNs = 0; };
	std::map<uint64_t, GFrame> gframes;
	std::vector<CUdeviceptr> bufPool;
	size_t compBytes = 0;
	std::atomic<int> outBpp{4};   // composite bytes/pixel: 4 = RGBA8, 8 = RGBA16 (HDR)

	// Latest-wins published wall (8 bpc, 4 ch, top-down, pitch = wallW*4).
	std::mutex latestMtx; std::condition_variable latestCv;
	std::vector<uint8_t> latestPx; RecvWallFrameInfo latestInfo{};
	std::atomic<unsigned long long> version{0};

	// GPU-direct output: registered D3D11 texture the GPU path publishes into
	// (device composite -> texture array, no CPU round-trip).
	std::mutex texMtx;
	CUgraphicsResource texRes = nullptr;
	std::atomic<bool> texBound{false};

	// GPU-direct output: a CUDA array (e.g. a Vulkan/GL texture mapped to CUDA)
	// the GPU path publishes into directly. Persistent (no map/unmap); the caller
	// owns the array's lifetime and rebinds a fresh one per frame if double-buffering.
	std::mutex zc_mtx;
	CUarray zc_arr = nullptr;
	std::atomic<bool> zc_bound{false};

	std::atomic<uint64_t> ausDone{0}, decOut{0}, complete{0}, qDrops{0}, gpuErrors{0};
	std::mutex srcMtx; char srcAddr[48] = {0};   // last datagram's source ip:port
	uint8_t yLut[256], cLut[256];

	// ---- cross-instance sync (all inert when cfg.syncGroup[0] == 0) --------
	SyncBoardConn board;
	bool syncOn = false;
	int syncDepth = 8;
	struct SyncedWall { std::vector<uint8_t> px; sync::SyncMeta meta; uint64_t sendTimeNs = 0; };
	std::mutex syncMtx; std::condition_variable syncCv;   // reorder + presenter wakeup
	std::map<uint64_t, SyncedWall> reorder;               // key = SEI globalFrameIndex
	std::vector<std::vector<uint8_t>> wallPool;           // recycled wall buffers
	uint64_t myLastComplete = SEI_NONE, myBits = 0;       // mirror of my slot
	uint64_t lastPublishedSei = SEI_NONE;
	uint64_t lastBeatTick = 0;                            // rx heartbeat throttle
	std::atomic<uint64_t> holdSinceTick{0};               // 0 = not holding
	std::atomic<uint64_t> bufferMisses{0}, epochResets{0};
	std::atomic<int> membersActiveStat{0}, membersGatingStat{0};
	std::thread presenter;

	void enqueueSynced(std::vector<uint8_t>&& wall, const sync::SyncMeta& m,
	                   uint64_t sendTimeNs, bool fixAlpha);
	void presenterLoop();

	void publish(std::vector<uint8_t>& wall, const sync::SyncMeta& m, uint64_t sendTimeNs,
	             bool fixAlpha = false)
	{
		// The SDK's Nv12ToColor32 kernel writes RGB and leaves alpha at 0; hosts
		// that honor alpha (Resolve float RGBA) composite that to black. Force
		// opaque here so every client gets A=255 on both convert paths.
		if (fixAlpha)
			for (size_t i = 3; i < wall.size(); i += 4) wall[i] = 255;
		{
			// version advances under latestMtx so RecvWallWaitNewFrame can't miss
			// the notify between its predicate check and going to sleep.
			std::lock_guard<std::mutex> lk(latestMtx);
			latestPx.swap(wall);
			fillInfo(latestInfo, m, sendTimeNs, wallW, wallH);
			version.fetch_add(1);
		}
		complete.fetch_add(1);
		latestCv.notify_all();
	}

	// First SyncMeta fixes the wall geometry (caller holds pendingMtx).
	//
	// gridCols/gridRows (uint16) and tileW/tileH (uint32) come off the wire, and
	// every allocation below is sized from their products. Narrowing tileW to int
	// and then computing cols*tileW in int overflowed on hostile or corrupt
	// values, so validate in 64-bit first and leave haveGeo false (no frames are
	// published, the producer keeps showing its marker) if anything is out of
	// range, rather than latching a geometry that overflows.
	static constexpr long long kMaxTileDim  = 16384;
	static constexpr long long kMaxGridDim  = 64;
	static constexpr long long kMaxWallDim  = 16384;
	static constexpr long long kMaxWallPx   = 64ll << 20; // 64 Mpx (~512 MB at 8 bpp)

	void configureLocked(const sync::SyncMeta& m)
	{
		if (cols) return;

		const long long c  = m.gridCols ? (long long)m.gridCols : 1;
		const long long r  = m.gridRows ? (long long)m.gridRows : 1;
		const long long tw = (long long)m.tileW;
		const long long th = (long long)m.tileH;

		if (c > kMaxGridDim || r > kMaxGridDim ||
		    tw < 1 || th < 1 || tw > kMaxTileDim || th > kMaxTileDim) {
			std::fprintf(stderr, "[recvwall] rejecting wall geometry: grid %lldx%lld tile %lldx%lld\n",
			             c, r, tw, th);
			return;
		}

		const long long ww = c * tw, wh = r * th;
		if (ww > kMaxWallDim || wh > kMaxWallDim || ww * wh > kMaxWallPx) {
			std::fprintf(stderr, "[recvwall] rejecting wall geometry: wall %lldx%lld\n", ww, wh);
			return;
		}

		cols = (int)c; rows = (int)r;
		tileW = (int)tw; tileH = (int)th;
		wallW = (int)ww; wallH = (int)wh;
		fpsNum = m.fpsNum; fpsDen = m.fpsDen ? m.fpsDen : 1;
		if (expectedTiles <= 0) expectedTiles = cols * rows;
		haveGeo.store(true);
	}

	void decodeWorkerCpu(uint16_t tile, TileQ* tq);
	void decodeWorkerGpu(uint16_t tile, TileQ* tq);
};

void RecvWallHandle::decodeWorkerCpu(uint16_t tile, TileQ* tq)
{
	cuCtxSetCurrent(ctx);
	NvDecoder dec(ctx, false, pickCudaCodec(cfg.codec), true);
	std::vector<sync::SyncMeta> metas;
	for (;;) {
		AuJob job;
		{ std::unique_lock<std::mutex> lk(tq->m);
		  tq->cv.wait(lk, [&] { return !running.load() || !tq->q.empty(); });
		  if (!running.load() && tq->q.empty()) break;
		  job = std::move(tq->q.front()); tq->q.pop_front(); }
		metas.clear();
		scanMeta(cfg.codec, job.au.data(), job.au.size(), [&](const sync::SyncMeta& m) { metas.push_back(m); });
		int n = dec.Decode(job.au.data(), (int)job.au.size());
		decOut.fetch_add((uint64_t)n);
		for (int i = 0; i < n; ++i) {
			uint8_t* fr = dec.GetFrame();
			if (!fr || metas.empty()) continue;
			const sync::SyncMeta& m = metas.back();
			std::vector<uint8_t> nv((size_t)dec.GetWidth() * dec.GetHeight() * 3 / 2);
			std::memcpy(nv.data(), fr, nv.size());
			// Insert under the lock; if this tile completes the wall frame, move the
			// slot out and composite OUTSIDE the lock.
			std::map<uint16_t, std::vector<uint8_t>> done;
			sync::SyncMeta doneMeta; uint64_t doneSend = 0;
			{ std::lock_guard<std::mutex> lk(pendingMtx);
			  configureLocked(m);
			  // Geometry rejected (see configureLocked): expectedTiles is still 0, so the
			  // composite below would run against a zero-sized wall. Drop the job.
			  if (!haveGeo.load()) continue;
			  // Sender restart: its transport counter rebases to 0 and every new
			  // wall would be rejected as stale by lastComplete. Large backward
			  // jump => reset the aligner's live edge.
			  { long long lc = lastComplete.load();
			    if (lc > 300 && (long long)job.frame + 300 < lc) {
			        pending.clear(); pendingInfo.clear();
			        lastComplete.store(-1);
			    } }
			  auto& slot = pending[job.frame];
			  slot[m.tileId] = std::move(nv);
			  pendingInfo[job.frame] = { m, job.sendTimeNs };
			  if ((int)slot.size() >= expectedTiles && (long long)job.frame > lastComplete.load()) {
			      done.swap(slot);
			      doneMeta = pendingInfo[job.frame].first; doneSend = pendingInfo[job.frame].second;
			      lastComplete.store((long long)job.frame);
			      pending.erase(pending.begin(), pending.upper_bound(job.frame));
			      pendingInfo.erase(pendingInfo.begin(), pendingInfo.upper_bound(job.frame));
			  } }
			if (!done.empty()) {
				std::vector<uint8_t> wall((size_t)wallW * wallH * 4, 0);
				const int decW = dec.GetWidth(), decH = dec.GetHeight();
				for (auto& tk : done) {
					int tid = tk.first, tc = tid % cols, tr = tid / cols;
					nv12ToWall(tk.second.data(), decW, decH, tileW, tileH, wall.data(), wallW * 4,
					           tc * tileW, tr * tileH, wallW, wallH, yLut, cLut, cfg.pixelOrder == 1,
					           matrixFor(doneMeta.colorSpace));
				}
				if (syncOn) enqueueSynced(std::move(wall), doneMeta, doneSend, false);
				else publish(wall, doneMeta, doneSend);
			}
		}
	}
}

void RecvWallHandle::decodeWorkerGpu(uint16_t tile, TileQ* tq)
{
	cuCtxSetCurrent(ctx);
	NvDecoder dec(ctx, true, pickCudaCodec(cfg.codec), true); // device frames
	std::vector<sync::SyncMeta> metas;
	for (;;) {
		AuJob job;
		{ std::unique_lock<std::mutex> lk(tq->m);
		  tq->cv.wait(lk, [&] { return !running.load() || !tq->q.empty(); });
		  if (!running.load() && tq->q.empty()) break;
		  job = std::move(tq->q.front()); tq->q.pop_front(); }
		metas.clear();
		scanMeta(cfg.codec, job.au.data(), job.au.size(), [&](const sync::SyncMeta& m) { metas.push_back(m); });
		int n = dec.Decode(job.au.data(), (int)job.au.size());
		decOut.fetch_add((uint64_t)n);
		for (int i = 0; i < n; ++i) {
			uint8_t* fr = dec.GetFrame();
			if (!fr || metas.empty()) continue;
			const sync::SyncMeta& m = metas.back();
			bool fresh = false; CUdeviceptr myBuf = 0; sync::SyncMeta doneMeta; uint64_t doneSend = 0;
			const bool p016 = dec.GetBitDepth() > 8;   // 10/12-bit HDR surface (P016)
			const int  bpp  = p016 ? 8 : 4;            // composite bytes/pixel (RGBA16 / RGBA8)
			{ std::lock_guard<std::mutex> lk(pendingMtx);
			  configureLocked(m);
			  // Geometry rejected (see configureLocked): nothing below can be sized. Drop it.
			  if (!haveGeo.load()) continue;
			  outBpp.store(bpp);
			  if (compBytes == 0) compBytes = (size_t)wallW * wallH * bpp;
			  { long long lc = lastComplete.load();   // sender restart (see CPU worker)
			    if (lc > 300 && (long long)job.frame + 300 < lc) {
			        for (auto& kv : gframes) if (kv.second.dComp) bufPool.push_back(kv.second.dComp);
			        gframes.clear();
			        lastComplete.store(-1);
			    } }
			  GFrame& gf = gframes[job.frame];
			  if (gf.dComp == 0) {
			      if (!bufPool.empty()) { gf.dComp = bufPool.back(); bufPool.pop_back(); }
			      else cuMemAlloc(&gf.dComp, compBytes);
			      gf.written = 0;
			  }
			  gf.meta = m; gf.sendTimeNs = job.sendTimeNs;
			  const int tc = m.tileId % cols, tr = m.tileId / cols;
			  const int roiW = tileW < dec.GetWidth() ? tileW : dec.GetWidth();
			  const int roiH = tileH < dec.GetHeight() ? tileH : dec.GetHeight();
			  const CUdeviceptr dst = gf.dComp + (CUdeviceptr)(((size_t)(tr * tileH) * wallW + (size_t)(tc * tileW)) * bpp);
			  if (p016) {
			      // 10/12-bit: NVDEC gives P016; convert to 16-bit RGBA. Pick the YUV->RGB
			      // matrix from the SEI colour tag (HDR walls are BT.2020).
			      const int mat = matrixFor(m.colorSpace);
			      if (cfg.pixelOrder == 1)
			          P016ToColor64<RGBA64>(fr, dec.GetDeviceFramePitch(), (uint8_t*)dst, wallW * bpp,
			                                roiW, roiH, mat, cfg.fullRange != 0);
			      else
			          P016ToColor64<BGRA64>(fr, dec.GetDeviceFramePitch(), (uint8_t*)dst, wallW * bpp,
			                                roiW, roiH, mat, cfg.fullRange != 0);
			  } else if (cfg.pixelOrder == 1) {
			      Nv12ToColor32<RGBA32>(fr, dec.GetDeviceFramePitch(), (uint8_t*)dst, wallW * bpp,
			                            roiW, roiH, matrixFor(m.colorSpace), cfg.fullRange != 0);
			  } else {
			      Nv12ToColor32<BGRA32>(fr, dec.GetDeviceFramePitch(), (uint8_t*)dst, wallW * bpp,
			                            roiW, roiH, matrixFor(m.colorSpace), cfg.fullRange != 0);
			  }
			  // Inside CUDA-heavy host processes (e.g. Resolve) the kernel launch can
			  // fail where standalone processes work; surface it so the client can
			  // fall back to the CPU convert instead of compositing black.
			  cudaError_t ce = cudaGetLastError();
			  if (ce != cudaSuccess) {
			      if (gpuErrors.fetch_add(1) == 0)
			          std::fprintf(stderr, "[recvwall] GPU convert kernel failed (%s) - "
			                       "re-init with useGpuConvert=0\n", cudaGetErrorString(ce));
			  }
			  if (++gf.written >= expectedTiles) {
			      fresh = (long long)job.frame > lastComplete.load();
			      myBuf = gf.dComp; doneMeta = gf.meta; doneSend = gf.sendTimeNs;
			      if (fresh) lastComplete.store((long long)job.frame);
			      // reclaim this + any older (stale) frame buffers
			      for (auto it = gframes.begin(); it != gframes.end() && it->first <= job.frame; ) {
			          if (it->second.dComp && it->second.dComp != myBuf) bufPool.push_back(it->second.dComp);
			          it = gframes.erase(it);
			      }
			  } }
			if (myBuf) {
				if (fresh) {
					cuCtxSynchronize();   // wait for all tile kernels before reading back
					bool wentDirect = false;
					// Zero-copy publish into a bound CUDA array (Vulkan/GL texture mapped
					// to CUDA): device composite -> array, entirely on the GPU. The array
					// is persistent (no map/unmap); clients use RecvWallPeekInfo.
					if (zc_bound.load() && !syncOn) {
						std::lock_guard<std::mutex> zc_lock(zc_mtx); // NB: 'ck' is a NvCodecUtils macro
						if (zc_arr) {
							if (bpp == 8)
								cuMemsetD2D16(myBuf + 6, 8, 0xFFFF, 1, (size_t)wallW * wallH); // 16-bit opaque alpha
							else
								cuMemsetD2D8(myBuf + 3, 4, 0xFF, 1, (size_t)wallW * wallH);   // 8-bit opaque alpha
							CUDA_MEMCPY2D cp{};
							cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
							cp.srcDevice     = myBuf;
							cp.srcPitch      = (size_t)wallW * bpp;
							cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
							cp.dstArray      = zc_arr;
							cp.WidthInBytes  = (size_t)wallW * bpp;
							cp.Height        = (size_t)wallH;
							if (cuMemcpy2D(&cp) == CUDA_SUCCESS) wentDirect = true;
						}
						if (wentDirect) {
							std::lock_guard<std::mutex> lk(latestMtx);
							fillInfo(latestInfo, doneMeta, doneSend, wallW, wallH);
							version.fetch_add(1);
							complete.fetch_add(1);
							latestCv.notify_all();
						}
					}
					if (!wentDirect && texBound.load() && !syncOn && bpp == 4) {
						// Zero-copy publish: CUDA composite -> registered D3D11
						// texture, entirely on the GPU. Version/info only; no
						// CPU pixels (clients use RecvWallPeekInfo). 8-bit only
						// (the bound texture is R8G8B8A8; HDR falls to readback).
						// The SDK convert kernel leaves alpha 0 (the CPU path
						// fixes it host-side): strided GPU memset -> opaque.
						cuMemsetD2D8(myBuf + 3, 4, 0xFF, 1, (size_t)wallW * wallH);
						std::lock_guard<std::mutex> tk(texMtx);
						if (texRes) {
							CUresult r = cuGraphicsMapResources(1, &texRes, 0);
							if (r == CUDA_SUCCESS) {
								CUarray arr = nullptr;
								if (cuGraphicsSubResourceGetMappedArray(&arr, texRes, 0, 0) == CUDA_SUCCESS) {
									CUDA_MEMCPY2D cp{};
									cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
									cp.srcDevice = myBuf;
									cp.srcPitch = (size_t)wallW * 4;
									cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
									cp.dstArray = arr;
									cp.WidthInBytes = (size_t)wallW * 4;
									cp.Height = (size_t)wallH;
									if (cuMemcpy2D(&cp) == CUDA_SUCCESS) wentDirect = true;
								}
								cuGraphicsUnmapResources(1, &texRes, 0);
							}
						}
						if (wentDirect) {
							std::lock_guard<std::mutex> lk(latestMtx);
							fillInfo(latestInfo, doneMeta, doneSend, wallW, wallH);
							version.fetch_add(1);
						}
						if (wentDirect) { complete.fetch_add(1); latestCv.notify_all(); }
					}
					if (!wentDirect) {
						std::vector<uint8_t> wall(compBytes);
						cuMemcpyDtoH(wall.data(), myBuf, compBytes);
						// The convert kernel leaves alpha 0; force opaque host-side
						// (depth-aware: 8-bit alpha every 4th byte, 16-bit every 8th).
						if (bpp == 8)
							for (size_t a = 6; a + 2 <= wall.size(); a += 8) { wall[a] = 0xFF; wall[a + 1] = 0xFF; }
						else
							for (size_t a = 3; a < wall.size(); a += 4) wall[a] = 0xFF;
						if (syncOn) enqueueSynced(std::move(wall), doneMeta, doneSend, /*fixAlpha=*/false);
						else publish(wall, doneMeta, doneSend, /*fixAlpha=*/false);
					}
				}
				std::lock_guard<std::mutex> lk(pendingMtx);
				bufPool.push_back(myBuf);
			}
		}
	}
}

// Completed wall -> reorder buffer keyed by SEI index + publish my slot on the
// board. The presenter decides what (and whether) to actually present.
void RecvWallHandle::enqueueSynced(std::vector<uint8_t>&& wall, const sync::SyncMeta& m,
                                   uint64_t sendTimeNs, bool fixAlpha)
{
	const uint64_t S = m.globalFrameIndex;   // SEI index: the cross-stream truth
	if (fixAlpha)
		for (size_t i = 3; i < wall.size(); i += 4) wall[i] = 255;

	bool epochReset = false;
	{
		std::lock_guard<std::mutex> lk(syncMtx);
		// Own index jumped backward beyond the bitmask window: sender restarted
		// (or a test stream looped). Old-epoch frames are unreachable now.
		if (myLastComplete != SEI_NONE && S + 64 < myLastComplete) {
			epochReset = true;
			for (auto& kv : reorder) wallPool.push_back(std::move(kv.second.px));
			reorder.clear();
			lastPublishedSei = SEI_NONE;
			myBits = 0; myLastComplete = SEI_NONE;
			epochResets.fetch_add(1);
		}
		if (myLastComplete == SEI_NONE || S > myLastComplete) {
			uint64_t sh = (myLastComplete == SEI_NONE) ? 64 : S - myLastComplete;
			myBits = (sh >= 64) ? 1ull : ((myBits << sh) | 1ull);
			myLastComplete = S;
		} else if (myLastComplete - S < 64) {
			myBits |= 1ull << (myLastComplete - S);   // late completion inside window
		}
		SyncedWall sw; sw.px = std::move(wall); sw.meta = m; sw.sendTimeNs = sendTimeNs;
		reorder[S] = std::move(sw);
		while ((int)reorder.size() > syncDepth) {
			wallPool.push_back(std::move(reorder.begin()->second.px));
			reorder.erase(reorder.begin());
		}
	}
	if (board.lock()) {
		SyncSlot& s = board.bd->slots[board.mySlot];
		const uint64_t now = GetTickCount64();
		if (epochReset) s.epochResets++;
		s.lastCompleteSei = myLastComplete;
		s.completedBits = myBits;
		s.progressTickMs = now;
		s.heartbeatTickMs = now;
		board.unlock();
	}
	syncCv.notify_one();
}

// Presenter: publish the highest SEI index every gating member has completed.
// Strict semantics fall out naturally: no common frame -> publish nothing ->
// latestPx/version freeze at the last common frame for every client.
void RecvWallHandle::presenterLoop()
{
	while (running.load()) {
		{
			std::unique_lock<std::mutex> lk(syncMtx);
			syncCv.wait_for(lk, std::chrono::milliseconds(2));   // 2 ms poll bounds
		}                                                        // cross-process skew
		if (!running.load()) break;

		SyncSlot snap[kSyncSlots];
		if (!board.lock()) continue;
		std::memcpy(snap, board.bd->slots, sizeof(snap));
		board.unlock();

		const uint64_t now = GetTickCount64();
		auto gates = [&](const SyncSlot& s) {
			if (!s.active) return false;
			if (s.lastCompleteSei == SEI_NONE) return false;      // JOINING
			if (cfg.syncTimeoutMs > 0 &&
			    now - s.progressTickMs > (uint64_t)cfg.syncTimeoutMs) return false;
			return true;
		};
		int nAct = 0, gating = 0;
		uint64_t hCand = SEI_NONE;
		for (auto& s : snap) {
			if (s.active) nAct++;
			if (gates(s)) { gating++; if (s.lastCompleteSei < hCand) hCand = s.lastCompleteSei; }
		}
		membersActiveStat.store(nAct);
		membersGatingStat.store(gating);

		uint64_t H = SEI_NONE;
		if (gating > 0 && hCand != SEI_NONE) {
			for (uint64_t x = hCand;; --x) {          // <=64 downward steps
				bool all = true;
				for (auto& s : snap) {
					if (!gates(s)) continue;
					uint64_t d = s.lastCompleteSei - x;   // x <= every lastComplete
					if (d >= 64 || !((s.completedBits >> d) & 1)) { all = false; break; }
				}
				if (all) { H = x; break; }
				if (x == 0 || hCand - x >= 63) break;
			}
		}

		std::vector<uint8_t> toPub; sync::SyncMeta pm{}; uint64_t ps = 0; bool doPub = false;
		{
			std::lock_guard<std::mutex> lk(syncMtx);
			const bool newDecoded = myLastComplete != SEI_NONE &&
				(lastPublishedSei == SEI_NONE || myLastComplete > lastPublishedSei);
			if (H != SEI_NONE && (lastPublishedSei == SEI_NONE || H > lastPublishedSei)) {
				auto it = reorder.find(H);
				if (it != reorder.end()) {
					toPub = std::move(it->second.px); pm = it->second.meta;
					ps = it->second.sendTimeNs; doPub = true;
				} else bufferMisses.fetch_add(1);     // evicted: depth < member skew
				lastPublishedSei = H;                 // advance regardless: never stick
				for (auto it2 = reorder.begin(); it2 != reorder.end() && it2->first <= H; ) {
					wallPool.push_back(std::move(it2->second.px));
					it2 = reorder.erase(it2);
				}
				holdSinceTick.store(0);
			} else if (newDecoded) {                  // frames waiting, no consensus
				uint64_t exp = 0;
				holdSinceTick.compare_exchange_strong(exp, now);
			} else {
				holdSinceTick.store(0);               // nothing new at all
			}
		}
		if (doPub) publish(toPub, pm, ps);            // alpha already fixed at enqueue
	}
}

extern "C" {

RecvWallHandle* RecvWallInit(const RecvWallConfig* cfg)
{
	if (!cfg) return nullptr;
	RecvWallHandle* h = new RecvWallHandle();
	h->cfg = *cfg;
	if (h->cfg.maxAuQueue <= 0) h->cfg.maxAuQueue = 16;
	h->expectedTiles = h->cfg.expectedTiles;

	// Range LUTs (decklink_player model): full->limited studio swing, or identity.
	for (int i = 0; i < 256; ++i) {
		h->yLut[i] = h->cfg.fullRange ? (uint8_t)i : (uint8_t)(16 + (i * 219 + 127) / 255);
		h->cLut[i] = h->cfg.fullRange ? (uint8_t)i : (uint8_t)(16 + (i * 224 + 127) / 255);
	}

	if (cuInit(0) != CUDA_SUCCESS) { delete h; return nullptr; }
	int cudaDev = h->cfg.cudaDevice;
	{ int nDev = 0; if (cuDeviceGetCount(&nDev) != CUDA_SUCCESS || cudaDev < 0 || cudaDev >= nDev) cudaDev = 0; }
	if (cuDeviceGet(&h->cuDev, cudaDev) != CUDA_SUCCESS) { delete h; return nullptr; }
	if (cuDevicePrimaryCtxRetain(&h->ctx, h->cuDev) != CUDA_SUCCESS) { delete h; return nullptr; }

	WSADATA W;
	if (WSAStartup(MAKEWORD(2, 2), &W) != 0) { cuDevicePrimaryCtxRelease(h->cuDev); delete h; return nullptr; }
	h->wsa = true;
	h->sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	int rb = (h->cfg.rcvbufKB > 0 ? h->cfg.rcvbufKB : 65536) << 10;
	setsockopt(h->sock, SOL_SOCKET, SO_RCVBUF, (char*)&rb, sizeof(rb));
	sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_ANY);
	if (h->cfg.bindIp[0] && inet_pton(AF_INET, h->cfg.bindIp, &a.sin_addr) != 1) {
		closesocket(h->sock); WSACleanup(); cuDevicePrimaryCtxRelease(h->cuDev); delete h; return nullptr;
	}
	a.sin_port = htons(h->cfg.listenPort ? h->cfg.listenPort : 9000);
	if (bind(h->sock, (sockaddr*)&a, sizeof(a)) != 0) {
		closesocket(h->sock); WSACleanup(); cuDevicePrimaryCtxRelease(h->cuDev); delete h; return nullptr;
	}
	DWORD to = 200; setsockopt(h->sock, SOL_SOCKET, SO_RCVTIMEO, (char*)&to, sizeof(to));

	// Reassembled AU -> per-tile bounded queue; lazily spawn one worker per tile.
	h->auCb = [h](uint16_t tile, uint32_t frame,
	              const std::vector<uint8_t>& au, uint64_t sendTimeNs, int) {
		// Diagnostic: RECVWALL_DUMP_AU=<prefix> dumps the first AU per tile to
		// <prefix>_t<tile>.bin for offline SEI/bitstream inspection.
		static char dumpPrefix[260] = {0};
		static std::atomic<int> dumpInit{0}, dumped{0};
		if (dumpInit.fetch_add(1) == 0) {
			DWORD n = GetEnvironmentVariableA("RECVWALL_DUMP_AU", dumpPrefix, sizeof(dumpPrefix));
			if (n == 0 || n >= sizeof(dumpPrefix)) dumpPrefix[0] = 0;
		}
		if (dumpPrefix[0] && dumped.load() < 4) {
			char path[300];
			std::snprintf(path, sizeof(path), "%s_t%u.bin", dumpPrefix, (unsigned)tile);
			FILE* f = std::fopen(path, "wxb");           // first AU per tile only
			if (f) { std::fwrite(au.data(), 1, au.size(), f); std::fclose(f); dumped.fetch_add(1); }
		}
		h->ausDone.fetch_add(1);
		RecvWallHandle::TileQ* tq = nullptr;
		{ std::lock_guard<std::mutex> lk(h->tqMtx);
		  auto it = h->tileQs.find(tile);
		  if (it == h->tileQs.end()) {
		      if (h->tileQs.size() >= 32) return;                       // sanity bound
		      tq = new RecvWallHandle::TileQ(); h->tileQs[tile] = tq;
		      if (h->cfg.useGpuConvert) h->workers.emplace_back(&RecvWallHandle::decodeWorkerGpu, h, tile, tq);
		      else                      h->workers.emplace_back(&RecvWallHandle::decodeWorkerCpu, h, tile, tq);
		  } else tq = it->second; }
		{ std::lock_guard<std::mutex> lk(tq->m);
		  if ((int)tq->q.size() >= h->cfg.maxAuQueue) { tq->q.pop_front(); h->qDrops.fetch_add(1); }
		  tq->q.push_back(RecvWallHandle::AuJob{ frame, sendTimeNs, au }); }
		tq->cv.notify_one();
	};
	h->reasm = new rtpfec::Reassembler(h->auCb);

	// Cross-instance sync: join the named board and start the presenter. On any
	// failure sync is disabled but the receiver stays fully functional.
	if (h->cfg.syncGroup[0]) {
		h->syncDepth = h->cfg.syncBufferFrames > 0 ? h->cfg.syncBufferFrames : 8;
		h->syncOn = h->board.open(h->cfg.syncGroup);
		if (h->syncOn) h->presenter = std::thread(&RecvWallHandle::presenterLoop, h);
		else std::fprintf(stderr, "[recvwall] sync group '%s' unavailable (board full?) - running unsynced\n",
		                  h->cfg.syncGroup);
	}

	h->rx = std::thread([h] {
		std::vector<uint8_t> buf(70000);
		sockaddr_in from{}; int fromLen = sizeof(from);
		uint32_t lastIp = 0; uint16_t lastPort = 0;
		while (h->running.load()) {
			// Board heartbeat: proves this process alive even while no walls
			// complete (slot-reclaim input; NOT the progress the timeout gates on).
			if (h->syncOn) {
				uint64_t now = GetTickCount64();
				if (now - h->lastBeatTick > 250 && h->board.lock()) {
					h->board.bd->slots[h->board.mySlot].heartbeatTickMs = now;
					h->board.unlock();
					h->lastBeatTick = now;
				}
			}
			fromLen = sizeof(from);
			int n = recvfrom(h->sock, (char*)buf.data(), (int)buf.size(), 0, (sockaddr*)&from, &fromLen);
			if (n > 0) {
				if (from.sin_addr.s_addr != lastIp || from.sin_port != lastPort) {
					lastIp = from.sin_addr.s_addr; lastPort = from.sin_port;
					char ip[32]; inet_ntop(AF_INET, &from.sin_addr, ip, sizeof(ip));
					std::lock_guard<std::mutex> lk(h->srcMtx);
					std::snprintf(h->srcAddr, sizeof(h->srcAddr), "%s:%u", ip, (unsigned)ntohs(from.sin_port));
				}
				// Sender restart: the transport counter rebases to 0 but the
				// Reassembler remembers completed frames forever ("done" markers),
				// so it would swallow every repeated index until passing the old
				// high-water mark. Rebuild it on a large backward jump.
				if (n >= (int)sizeof(rtpfec::PktHdr)) {
					rtpfec::PktHdr ph; std::memcpy(&ph, buf.data(), sizeof(ph));
					if (ph.magic == rtpfec::kMagic) {
						if (h->rxHwm > 300 && ph.frameIndex + 300 < h->rxHwm) {
							std::lock_guard<std::mutex> lk(h->reasmMtx);
							delete h->reasm;
							h->reasm = new rtpfec::Reassembler(h->auCb);
							h->rxHwm = 0;
							std::fprintf(stderr, "[recvwall] sender restart detected - reassembler reset\n");
						}
						if (ph.frameIndex > h->rxHwm) h->rxHwm = ph.frameIndex;
					}
				}
				h->reasm->onPacket(buf.data(), n);
			}
		}
	});
	return h;
}

int RecvWallGetGeometry(RecvWallHandle* h, unsigned int* wallW, unsigned int* wallH,
                        unsigned short* gridCols, unsigned short* gridRows,
                        unsigned int* fpsNum, unsigned int* fpsDen)
{
	if (!h || !h->haveGeo.load()) return 0;
	std::lock_guard<std::mutex> lk(h->pendingMtx);
	if (wallW) *wallW = (unsigned)h->wallW;
	if (wallH) *wallH = (unsigned)h->wallH;
	if (gridCols) *gridCols = (unsigned short)h->cols;
	if (gridRows) *gridRows = (unsigned short)h->rows;
	if (fpsNum) *fpsNum = h->fpsNum;
	if (fpsDen) *fpsDen = h->fpsDen;
	return 1;
}

int RecvWallGetPixelBytes(RecvWallHandle* h) { return h ? h->outBpp.load() : 4; }

int RecvWallGetLatest(RecvWallHandle* h, unsigned char* dst, int dstPitch,
                      int dstCapBytes, unsigned long long* ioVersion,
                      RecvWallFrameInfo* outInfo)
{
	if (!h || !dst || !ioVersion || !h->haveGeo.load()) return -1;
	unsigned long long v = h->version.load();
	if (v == *ioVersion) return 0;
	std::lock_guard<std::mutex> lk(h->latestMtx);
	if (h->latestPx.empty()) return 0;
	const int srcPitch = h->wallW * h->outBpp.load();
	if (dstPitch < srcPitch || (long long)dstCapBytes < (long long)dstPitch * h->wallH) return -1;
	for (int y = 0; y < h->wallH; ++y)
		std::memcpy(dst + (size_t)y * dstPitch, h->latestPx.data() + (size_t)y * srcPitch, srcPitch);
	if (outInfo) *outInfo = h->latestInfo;
	*ioVersion = h->version.load();
	return 1;
}

int RecvWallWaitNewFrame(RecvWallHandle* h, unsigned long long curVersion, int timeoutMs)
{
	if (!h) return 0;
	std::unique_lock<std::mutex> lk(h->latestMtx);
	return h->latestCv.wait_for(lk, std::chrono::milliseconds(timeoutMs > 0 ? timeoutMs : 0),
	                            [&] { return h->version.load() > curVersion; }) ? 1 : 0;
}

void RecvWallGetStats(RecvWallHandle* h, RecvWallStats* out)
{
	if (!h || !out) return;
	std::lock_guard<std::mutex> lk(h->reasmMtx);
	const rtpfec::Reassembler::Stats& rs = h->reasm->stats();
	out->pktData = rs.dataRecv; out->pktFec = rs.fecRecv; out->fecRecovered = rs.recovered;
	out->ausDone = h->ausDone.load(); out->framesDecoded = h->decOut.load();
	out->wallsComposited = h->complete.load(); out->auQueueDrops = h->qDrops.load();
	out->gpuConvertErrors = h->gpuErrors.load();
}

int RecvWallBindD3D11Texture(RecvWallHandle* h, void* d3d11Texture2D)
{
	if (!h || !d3d11Texture2D) return 0;
	if (!h->cfg.useGpuConvert || h->syncOn) return 0;   // needs the device composite
	cuCtxSetCurrent(h->ctx);
	CUgraphicsResource res = nullptr;
	if (cuGraphicsD3D11RegisterResource(&res, (ID3D11Resource*)d3d11Texture2D,
	                                    CU_GRAPHICS_REGISTER_FLAGS_NONE) != CUDA_SUCCESS) {
		std::fprintf(stderr, "[recvwall] cuGraphicsD3D11RegisterResource failed (adapter mismatch?)\n");
		return 0;
	}
	{
		std::lock_guard<std::mutex> lk(h->texMtx);
		if (h->texRes) cuGraphicsUnregisterResource(h->texRes);
		h->texRes = res;
	}
	h->texBound.store(true);
	return 1;
}

void RecvWallUnbindD3D11Texture(RecvWallHandle* h)
{
	if (!h) return;
	h->texBound.store(false);
	cuCtxSetCurrent(h->ctx);
	std::lock_guard<std::mutex> lk(h->texMtx);
	if (h->texRes) { cuGraphicsUnregisterResource(h->texRes); h->texRes = nullptr; }
}

int RecvWallBindCudaArray(RecvWallHandle* h, void* cudaArray)
{
	if (!h || !cudaArray) return 0;
	if (!h->cfg.useGpuConvert || h->syncOn) return 0;   // needs the device composite
	std::lock_guard<std::mutex> lk(h->zc_mtx);
	h->zc_arr = reinterpret_cast<CUarray>(cudaArray);  // cudaArray_t and CUarray alias the same object
	h->zc_bound.store(true);
	return 1;
}

void RecvWallUnbindCudaArray(RecvWallHandle* h)
{
	if (!h) return;
	h->zc_bound.store(false);
	std::lock_guard<std::mutex> lk(h->zc_mtx);
	h->zc_arr = nullptr;
}

int RecvWallPeekInfo(RecvWallHandle* h, unsigned long long* ioVersion, RecvWallFrameInfo* outInfo)
{
	if (!h || !ioVersion) return 0;
	unsigned long long v = h->version.load();
	if (v == *ioVersion) return 0;
	std::lock_guard<std::mutex> lk(h->latestMtx);
	if (outInfo) *outInfo = h->latestInfo;
	*ioVersion = h->version.load();
	return 1;
}

int RecvWallGetSyncState(RecvWallHandle* h, RecvWallSyncState* out)
{
	if (!out) return 0;
	std::memset(out, 0, sizeof(*out));
	out->lastPublishedSei = out->lastCompletedSei = SEI_NONE;
	if (!h || !h->cfg.syncGroup[0]) return 0;
	out->enabled = h->syncOn ? 1 : 0;
	out->membersActive = h->membersActiveStat.load();
	out->membersGating = h->membersGatingStat.load();
	uint64_t hs = h->holdSinceTick.load();
	out->holding = hs ? 1 : 0;
	out->holdMs = hs ? (unsigned)(GetTickCount64() - hs) : 0;
	out->bufferMisses = h->bufferMisses.load();
	out->epochResets = h->epochResets.load();
	{
		std::lock_guard<std::mutex> lk(h->syncMtx);
		out->lastPublishedSei = h->lastPublishedSei;
		out->lastCompletedSei = h->myLastComplete;
		out->bufferedFrames = (unsigned)h->reorder.size();
	}
	return 1;
}

void RecvWallGetSource(RecvWallHandle* h, char* buf, int cap)
{
	if (!buf || cap <= 0) return;
	buf[0] = 0;
	if (!h) return;
	std::lock_guard<std::mutex> lk(h->srcMtx);
	std::snprintf(buf, (size_t)cap, "%s", h->srcAddr);
}

void RecvWallShutdown(RecvWallHandle* h)
{
	if (!h) return;
	h->running.store(false);
	if (h->rx.joinable()) h->rx.join();          // recv() wakes on the 200 ms timeout
	h->syncCv.notify_all();
	if (h->presenter.joinable()) h->presenter.join();
	{ std::lock_guard<std::mutex> lk(h->tqMtx); for (auto& kv : h->tileQs) kv.second->cv.notify_all(); }
	for (auto& w : h->workers) if (w.joinable()) w.join();
	{ std::lock_guard<std::mutex> lk(h->tqMtx); for (auto& kv : h->tileQs) delete kv.second; h->tileQs.clear(); }
	if (h->sock != INVALID_SOCKET) closesocket(h->sock);
	if (h->wsa) WSACleanup();
	cuCtxSetCurrent(h->ctx);
	h->texBound.store(false);
	{ std::lock_guard<std::mutex> lk(h->texMtx);
	  if (h->texRes) { cuGraphicsUnregisterResource(h->texRes); h->texRes = nullptr; } }
	for (auto d : h->bufPool) cuMemFree(d);
	for (auto& kv : h->gframes) if (kv.second.dComp) cuMemFree(kv.second.dComp);
	{ std::lock_guard<std::mutex> lk(h->reasmMtx); delete h->reasm; h->reasm = nullptr; }
	if (h->syncOn) h->board.close();             // release my slot under the mutex
	cuDevicePrimaryCtxRelease(h->cuDev);
	delete h;
}

} // extern "C"
