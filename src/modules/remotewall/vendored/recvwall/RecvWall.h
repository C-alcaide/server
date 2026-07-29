// Pure C API for the tile-wall RECEIVER: UDP/RTP+FEC reassembly -> per-tile
// NVDEC decode -> alignment by the in-band SyncMeta globalFrameIndex ->
// latest-wins composited wall frame in host RAM (BGRA8 or RGBA8).
//
// Decode-side mirror of cloudXR's nvenc_tile.lib (src/rt/NvTile.h): compiled
// into a standalone static library (recv_wall.lib) with plain MSVC so the
// NVIDIA Video Codec SDK / CUDA / Winsock headers never enter the host's
// translation units. The OpenFX plugin (and any other client: Spout bridge,
// test CLI) includes ONLY this header and links the library.
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct RecvWallHandle RecvWallHandle;

typedef struct RecvWallConfig
{
	unsigned short listenPort;    /* UDP port to bind (senders default to 9000) */
	int expectedTiles;            /* 0 = auto from the first SyncMeta (gridCols*gridRows) */
	int codec;                    /* 0 = HEVC (default), 1 = H.264 */
	int fullRange;                /* 1 = full-range passthrough (default; hosts like OFX
	                                 want full-swing RGB); 0 = full->limited LUT on Y/C
	                                 before the RGB convert (SDI-style studio swing) */
	int pixelOrder;               /* 0 = BGRA, 1 = RGBA (OFX wants RGBA) */
	int useGpuConvert;            /* 1 = CUDA Nv12ToColor32 into a device composite +
	                                 one DtoH per wall frame; 0 = CPU convert (default) */
	int rcvbufKB;                 /* SO_RCVBUF in KB; 0 => 65536 (64 MB) */
	int maxAuQueue;               /* per-tile AU queue bound, drop-oldest; 0 => 16 */
	char bindIp[64];              /* local interface to bind ("" => 0.0.0.0 = all) */
	int cudaDevice;               /* CUDA device ordinal to decode/composite on; 0 =
	                                 first GPU (default). Set to the host mixer's GPU
	                                 so the zero-copy composite and the imported
	                                 texture live on ONE device (multi-GPU boxes). */

	/* ---- Cross-instance presentation sync (multi-zone walls) ---------------
	   Instances (any mix of OFX plugin instances - possibly in separate host
	   processes - Spout bridges and CLIs) that join the same named group
	   present frame-locked: each publishes only the highest SEI
	   globalFrameIndex that EVERY member of the group has completed. */
	char syncGroup[32];           /* "" = sync off (default; exact legacy behavior) */
	int  syncBufferFrames;        /* reorder-buffer depth in frames; 0 => 8. Must
	                                 exceed the max start-offset between streams. */
	int  syncTimeoutMs;           /* 0 = STRICT: hold forever for stalled members;
	                                 >0 = drop members whose progress is staler
	                                 than this many ms (availability mode) */
} RecvWallConfig;

/* SyncMeta of the wall frame handed out by RecvWallGetLatest. */
typedef struct RecvWallFrameInfo
{
	unsigned long long globalFrameIndex;
	unsigned long long createTimeNsUtc;   /* engine stamp (ns since Unix epoch) */
	unsigned long long sendTimeNs;        /* sender fragmentation stamp */
	unsigned int  wallW, wallH, tileW, tileH;
	unsigned short gridCols, gridRows;
	unsigned int  fpsNum, fpsDen;
	char colorSpace[16];                  /* free text, e.g. "BT709" */
	unsigned char tcHours, tcMinutes, tcSeconds, tcFrames, tcDrop, tcValid;
	float cam[62];                        /* CameraMeta, same packed-float layout */
} RecvWallFrameInfo;

typedef struct RecvWallStats
{
	unsigned long long pktData, pktFec, fecRecovered;   /* transport */
	unsigned long long ausDone, framesDecoded;          /* reassembled AUs / NVDEC outputs */
	unsigned long long wallsComposited, auQueueDrops;   /* complete walls / live-edge drops */
	unsigned long long gpuConvertErrors;                /* failed NV12->RGBA kernel launches
	                                                       (seen inside CUDA-heavy hosts like
	                                                       Resolve); >0 => client should
	                                                       re-init with useGpuConvert=0 */
} RecvWallStats;

/* Start the rx thread + decode workers. Returns NULL on failure (port bind,
   CUDA init). One handle = one bound UDP port. */
RecvWallHandle* RecvWallInit(const RecvWallConfig* cfg);

/* Wall geometry, known once the first SyncMeta arrives. Returns 0 until then. */
int RecvWallGetGeometry(RecvWallHandle* h, unsigned int* wallW, unsigned int* wallH,
                        unsigned short* gridCols, unsigned short* gridRows,
                        unsigned int* fpsNum, unsigned int* fpsDen);

/* Composite bytes/pixel of the published wall: 4 = RGBA8/BGRA8 (8-bit), 8 =
   RGBA16/BGRA16 (10/12-bit HDR, from a P016 NVDEC surface). Defaults to 4 until
   the first frame is decoded; clients size their target texture accordingly. */
int RecvWallGetPixelBytes(RecvWallHandle* h);

/* Non-blocking. If the newest composite's version differs from *ioVersion, copy
   it (8 bpc, 4 ch, top-down, dstPitch bytes/row) into dst and update *ioVersion.
   Returns 1 = new frame copied, 0 = no new frame, -1 = no geometry yet or
   dst too small (needs wallH*dstPitch, dstPitch >= wallW*4). */
int RecvWallGetLatest(RecvWallHandle* h, unsigned char* dst, int dstPitch,
                      int dstCapBytes, unsigned long long* ioVersion,
                      RecvWallFrameInfo* outInfo);

/* Block up to timeoutMs for the composite version to advance past curVersion.
   Returns 1 = a newer frame is available, 0 = timeout. */
int RecvWallWaitNewFrame(RecvWallHandle* h, unsigned long long curVersion, int timeoutMs);

void RecvWallGetStats(RecvWallHandle* h, RecvWallStats* out);

/* ---- GPU-direct output (zero-copy to a D3D11 texture) ---------------------
   Bind a D3D11 texture (same adapter as CUDA device 0, size == wall size,
   DXGI_FORMAT_R8G8B8A8_UNORM) and the GPU-convert path publishes by copying
   the CUDA composite STRAIGHT INTO IT (no CPU round-trip; RecvWallGetLatest
   stops receiving pixels). Requires useGpuConvert=1 and no syncGroup (the
   sync reorder buffer is host-side). Returns 1 on success. */
int RecvWallBindD3D11Texture(RecvWallHandle* h, void* d3d11Texture2D);

/* Detach the texture: publishes go back to the CPU path (GetLatest works again). */
void RecvWallUnbindD3D11Texture(RecvWallHandle* h);

/* ---- GPU-direct output (zero-copy to a CUDA array) -----------------------
   Bind a CUDA array (cudaArray_t / CUarray, size == wall size, RGBA8/BGRA8 to
   match pixelOrder) that the GPU-convert path publishes the composite INTO
   directly (device->array, no CPU round-trip). Intended for a Vulkan/GL texture
   already mapped to CUDA (e.g. an exportable VkImage imported via external
   memory). Requires useGpuConvert=1 and no syncGroup. Returns 1 on success.
   RecvWallPeekInfo advances the version when a new wall has been written. */
int RecvWallBindCudaArray(RecvWallHandle* h, void* cudaArray);

/* Detach the CUDA array: publishes go back to the CPU path (GetLatest works). */
void RecvWallUnbindCudaArray(RecvWallHandle* h);

/* Like RecvWallGetLatest but WITHOUT pixels: advances *ioVersion and fills
   outInfo when a newer wall was published. The polling companion of the
   GPU-direct mode. Returns 1 = newer, 0 = no change. */
int RecvWallPeekInfo(RecvWallHandle* h, unsigned long long* ioVersion,
                     RecvWallFrameInfo* outInfo);

/* State of the cross-instance sync group (all zero/enabled=0 when sync off). */
typedef struct RecvWallSyncState
{
	int enabled;                          /* board mapped and slot claimed */
	int membersActive;                    /* active slots incl. self */
	int membersGating;                    /* members currently gating the lock
	                                         (excl. JOINING / timeout-dropped) */
	int holding;                          /* 1 = newer walls decoded but no common
	                                         frame exists yet (strict hold) */
	unsigned int holdMs;                  /* duration of the current hold */
	unsigned long long lastPublishedSei;  /* ~0 until first publish */
	unsigned long long lastCompletedSei;  /* my decode head; ~0 until first wall */
	unsigned int bufferedFrames;          /* reorder-buffer occupancy */
	unsigned long long bufferMisses;      /* common frame already evicted (raise
	                                         syncBufferFrames) */
	unsigned long long epochResets;       /* own-index backward jumps handled */
} RecvWallSyncState;

/* Returns 0 when sync is disabled for this handle, 1 otherwise. */
int RecvWallGetSyncState(RecvWallHandle* h, RecvWallSyncState* out);

/* Source address (ip:port) of the most recent datagram, "" until one arrives.
   Useful on multi-homed hosts to confirm WHICH sender is feeding us. */
void RecvWallGetSource(RecvWallHandle* h, char* buf, int cap);

/* Stop all threads and release CUDA/socket resources. */
void RecvWallShutdown(RecvWallHandle* h);

#ifdef __cplusplus
}
#endif
