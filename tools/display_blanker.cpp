// display_blanker.exe — Desktop blanker companion for CasparCG GPU outputs.
//
// Creates persistent black TOPMOST windows on selected monitors. Sits between
// the Windows desktop and CasparCG's output windows. If CasparCG crashes or
// restarts, the blanker keeps the outputs black.
//
// For Pro-tier NVIDIA GPUs (Quadro/RTX A-series), can also acquire "dedicated
// displays" via NvAPI. Dedicated displays are detached from the Windows desktop
// at the driver level — the output stays black even if the blanker itself exits.
// This provides crash-proof blanking without needing a running process.
//
// Features:
//   - System tray icon with right-click menu
//   - Configuration window to select which monitors to blank
//   - NvAPI dedicated display acquisition (Pro GPUs, driver-level blanking)
//   - Enable/Disable blanking toggle
//   - "Start with Windows" option (adds to HKCU Run registry)
//   - Settings saved to display_blanker.ini next to the exe
//   - Single-instance enforcement via named mutex
//   - No console window (Windows subsystem)
//
// Usage:
//   display_blanker.exe                  Open config on first run, tray only after
//   display_blanker.exe --autostart      Start minimized from saved config
//   display_blanker.exe --match <str>    CLI: blank matching monitors
//   display_blanker.exe --all            CLI: blank all non-primary monitors
//
// Copyright (c) 2026 CasparCG Contributors. Licensed under GPLv3.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#include <commctrl.h>

#include <algorithm>
#include <string>
#include <vector>

#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "advapi32.lib")

// Icon resource ID (must match .rc file)
#define IDI_APP_ICON 101

// ─── Constants ──────────────────────────────────────────────────────────

static const UINT WM_TRAYICON       = WM_APP + 1;
static const UINT IDM_CONFIGURE     = 2001;
static const UINT IDM_TOGGLE        = 2002;
static const UINT IDM_EXIT          = 2003;
static const UINT IDC_SAVE_BTN      = 3001;
static const UINT IDC_CLOSE_BTN     = 3002;
static const UINT IDC_AUTOSTART_CHK = 3003;
static const UINT IDC_ENABLE_CHK    = 3004;
static const UINT IDC_MONITOR_BASE  = 4000;
static const UINT IDC_DEDICATED_LABEL = 5000;
static const UINT IDC_DEDICATED_BASE  = 5001; // checkboxes for dedicated displays
static const wchar_t* MUTEX_NAME    = L"CasparCG_DisplayBlanker_SingleInstance";
static const wchar_t* REGISTRY_KEY  = L"Software\\Microsoft\\Windows\\CurrentVersion\\Run";
static const wchar_t* REGISTRY_VAL  = L"CasparCG Display Blanker";

// ─── NvAPI dynamic loader (no SDK dependency) ───────────────────────────
//
// We load nvapi64.dll at runtime and resolve the few functions we need via
// NvAPI_QueryInterface. This avoids requiring the NvAPI SDK to build the
// blanker tool, and gracefully degrades on non-NVIDIA systems.

// NvAPI status codes we care about
using NvAPI_Status = int;
static const NvAPI_Status NVAPI_OK = 0;

// Minimal struct matching NV_MANAGED_DEDICATED_DISPLAY_INFO_V1
#pragma pack(push, 8)
struct NvDedicatedDisplayInfo
{
    uint32_t version;
    uint32_t displayId;
    uint32_t isAcquired : 1;
    uint32_t isMosaic   : 1;
    uint32_t reserved   : 30;
};
#pragma pack(pop)

// MAKE_NVAPI_VERSION macro equivalent: (sizeof(struct) | (ver << 16))
static constexpr uint32_t NVDDI_VER1 =
    static_cast<uint32_t>(sizeof(NvDedicatedDisplayInfo)) | (1u << 16);

// NvAPI function IDs from nvapi_interface.h (stable across driver versions)
static const uint32_t NVAPI_ID_INITIALIZE                    = 0x0150E828;
static const uint32_t NVAPI_ID_UNLOAD                        = 0xD22BDD7E;
static const uint32_t NVAPI_ID_GET_ERROR_MESSAGE             = 0x6C2D048C;
static const uint32_t NVAPI_ID_GET_MANAGED_DEDICATED_DISPLAYS = 0xDBDF0CB2;
static const uint32_t NVAPI_ID_ACQUIRE_DEDICATED_DISPLAY     = 0x47C917BA;
static const uint32_t NVAPI_ID_RELEASE_DEDICATED_DISPLAY     = 0x1247825F;

// Function pointer types
using PFN_NvAPI_QueryInterface                   = void*(*)(uint32_t);
using PFN_NvAPI_Initialize                       = NvAPI_Status(*)();
using PFN_NvAPI_Unload                           = NvAPI_Status(*)();
using PFN_NvAPI_GetErrorMessage                  = NvAPI_Status(*)(NvAPI_Status, char[64]);
using PFN_NvAPI_DISP_GetNvManagedDedicatedDisplays = NvAPI_Status(*)(uint32_t*, NvDedicatedDisplayInfo*);
using PFN_NvAPI_DISP_AcquireDedicatedDisplay     = NvAPI_Status(*)(uint32_t, uint64_t*);
using PFN_NvAPI_DISP_ReleaseDedicatedDisplay     = NvAPI_Status(*)(uint32_t);

// Resolved function pointers (null if NvAPI unavailable)
static PFN_NvAPI_Initialize                          g_nvapi_init     = nullptr;
static PFN_NvAPI_Unload                              g_nvapi_unload   = nullptr;
static PFN_NvAPI_GetErrorMessage                     g_nvapi_err_msg  = nullptr;
static PFN_NvAPI_DISP_GetNvManagedDedicatedDisplays  g_nvapi_get_dd   = nullptr;
static PFN_NvAPI_DISP_AcquireDedicatedDisplay        g_nvapi_acquire  = nullptr;
static PFN_NvAPI_DISP_ReleaseDedicatedDisplay        g_nvapi_release  = nullptr;
static bool                                          g_nvapi_loaded   = false;

static std::wstring nvapi_error_string(NvAPI_Status status)
{
    if (g_nvapi_err_msg) {
        char buf[64] = {};
        g_nvapi_err_msg(status, buf);
        return std::wstring(buf, buf + strlen(buf));
    }
    return L"error " + std::to_wstring(status);
}

static bool load_nvapi()
{
    HMODULE lib = LoadLibraryW(L"nvapi64.dll");
    if (!lib) return false;

    auto query = reinterpret_cast<PFN_NvAPI_QueryInterface>(
        GetProcAddress(lib, "nvapi_QueryInterface"));
    if (!query) { FreeLibrary(lib); return false; }

    g_nvapi_init    = reinterpret_cast<PFN_NvAPI_Initialize>(query(NVAPI_ID_INITIALIZE));
    g_nvapi_unload  = reinterpret_cast<PFN_NvAPI_Unload>(query(NVAPI_ID_UNLOAD));
    g_nvapi_err_msg = reinterpret_cast<PFN_NvAPI_GetErrorMessage>(query(NVAPI_ID_GET_ERROR_MESSAGE));
    g_nvapi_get_dd  = reinterpret_cast<PFN_NvAPI_DISP_GetNvManagedDedicatedDisplays>(query(NVAPI_ID_GET_MANAGED_DEDICATED_DISPLAYS));
    g_nvapi_acquire = reinterpret_cast<PFN_NvAPI_DISP_AcquireDedicatedDisplay>(query(NVAPI_ID_ACQUIRE_DEDICATED_DISPLAY));
    g_nvapi_release = reinterpret_cast<PFN_NvAPI_DISP_ReleaseDedicatedDisplay>(query(NVAPI_ID_RELEASE_DEDICATED_DISPLAY));

    if (!g_nvapi_init || !g_nvapi_get_dd || !g_nvapi_acquire || !g_nvapi_release)
        return false;

    auto status = g_nvapi_init();
    if (status != NVAPI_OK)
        return false;

    g_nvapi_loaded = true;
    return true;
}

// ─── Dedicated display tracking ─────────────────────────────────────────

struct dedicated_display
{
    uint32_t display_id   = 0;
    bool     is_acquired  = false;   // Acquired by another process?
    bool     is_mosaic    = false;
    bool     we_own       = false;   // We acquired it in this session
    bool     checked      = false;   // User wants us to acquire it
};

static std::vector<dedicated_display> g_dedicated;

static void enumerate_dedicated_displays()
{
    // Release any we currently own before re-enumerating
    for (auto& d : g_dedicated) {
        if (d.we_own && g_nvapi_release) {
            g_nvapi_release(d.display_id);
            d.we_own = false;
        }
    }

    auto old_checked = std::move(g_dedicated);
    g_dedicated.clear();

    if (!g_nvapi_loaded || !g_nvapi_get_dd)
        return;

    uint32_t count = 0;
    auto status = g_nvapi_get_dd(&count, nullptr);
    if (status != NVAPI_OK || count == 0)
        return;

    std::vector<NvDedicatedDisplayInfo> infos(count);
    for (auto& d : infos)
        d.version = NVDDI_VER1;

    status = g_nvapi_get_dd(&count, infos.data());
    if (status != NVAPI_OK)
        return;

    for (uint32_t i = 0; i < count; ++i) {
        dedicated_display dd;
        dd.display_id  = infos[i].displayId;
        dd.is_acquired = infos[i].isAcquired != 0;
        dd.is_mosaic   = infos[i].isMosaic != 0;
        dd.checked     = false;

        // Preserve checked state from previous enumeration
        for (const auto& o : old_checked) {
            if (o.display_id == dd.display_id) {
                dd.checked = o.checked;
                break;
            }
        }

        g_dedicated.push_back(dd);
    }
}

static void acquire_dedicated_displays()
{
    if (!g_nvapi_loaded || !g_nvapi_acquire)
        return;

    for (auto& d : g_dedicated) {
        if (d.checked && !d.we_own && !d.is_acquired) {
            uint64_t handle = 0;
            auto status = g_nvapi_acquire(d.display_id, &handle);
            if (status == NVAPI_OK) {
                d.we_own      = true;
                d.is_acquired = true;
            }
        } else if (!d.checked && d.we_own) {
            if (g_nvapi_release) {
                g_nvapi_release(d.display_id);
                d.we_own      = false;
                d.is_acquired = false;
            }
        }
    }
}

static void release_all_dedicated_displays()
{
    if (!g_nvapi_loaded || !g_nvapi_release)
        return;

    for (auto& d : g_dedicated) {
        if (d.we_own) {
            g_nvapi_release(d.display_id);
            d.we_own      = false;
            d.is_acquired = false;
        }
    }
}

// ─── Monitor info ───────────────────────────────────────────────────────

struct monitor_info
{
    std::wstring device_name;
    std::wstring adapter_name;
    std::wstring monitor_name;
    std::wstring monitor_id;
    RECT         rect;
    bool         is_primary;
    bool         checked;
};

// ─── Globals ────────────────────────────────────────────────────────────

static HINSTANCE                  g_instance       = nullptr;
static HICON                      g_app_icon       = nullptr;
static NOTIFYICONDATAW            g_nid            = {};
static HWND                       g_tray_hwnd      = nullptr;
static HWND                       g_config_hwnd    = nullptr;
static HBRUSH                     g_black_brush    = nullptr;
static HANDLE                     g_mutex          = nullptr;
static bool                       g_enabled        = true;
static bool                       g_autostart      = false;
static std::vector<monitor_info>  g_monitors;
static std::vector<HWND>          g_blanker_windows;
static std::wstring               g_ini_path;
static std::wstring               g_exe_path;

// Forward declarations
static void create_blanker_windows();
static void destroy_blanker_windows();
static void update_tray_tooltip();
static void save_settings();
static void show_config_window();
static bool show_confirmation_countdown();

// ─── Snapshot for revert ────────────────────────────────────────────────

struct settings_snapshot
{
    bool                         enabled;
    std::vector<bool>            monitor_checked;
    std::vector<bool>            dedicated_checked;
};

static settings_snapshot take_snapshot()
{
    settings_snapshot s;
    s.enabled = g_enabled;
    for (const auto& m : g_monitors)
        s.monitor_checked.push_back(m.checked);
    for (const auto& d : g_dedicated)
        s.dedicated_checked.push_back(d.checked);
    return s;
}

static void restore_snapshot(const settings_snapshot& s)
{
    g_enabled = s.enabled;
    for (size_t i = 0; i < g_monitors.size() && i < s.monitor_checked.size(); ++i)
        g_monitors[i].checked = s.monitor_checked[i];
    for (size_t i = 0; i < g_dedicated.size() && i < s.dedicated_checked.size(); ++i)
        g_dedicated[i].checked = s.dedicated_checked[i];
}

// ─── Utility ────────────────────────────────────────────────────────────

static bool icontains(const std::wstring& haystack, const std::wstring& needle)
{
    if (needle.empty()) return true;
    auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(),
                          [](wchar_t a, wchar_t b) { return towlower(a) == towlower(b); });
    return it != haystack.end();
}

static std::wstring get_exe_directory()
{
    wchar_t buf[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring path(buf);
    auto pos = path.find_last_of(L"\\/");
    return (pos != std::wstring::npos) ? path.substr(0, pos + 1) : L"";
}

// ─── INI file ───────────────────────────────────────────────────────────

static void save_settings()
{
    WritePrivateProfileStringW(L"General", L"Enabled",
                               g_enabled ? L"1" : L"0", g_ini_path.c_str());

    WritePrivateProfileSectionW(L"Monitors", L"\0", g_ini_path.c_str());
    int idx = 0;
    for (const auto& m : g_monitors) {
        if (m.checked) {
            auto key = L"Output" + std::to_wstring(idx++);
            WritePrivateProfileStringW(L"Monitors", key.c_str(),
                                       m.device_name.c_str(), g_ini_path.c_str());
        }
    }
    WritePrivateProfileStringW(L"Monitors", L"Count",
                               std::to_wstring(idx).c_str(), g_ini_path.c_str());

    // Save dedicated display selections
    WritePrivateProfileSectionW(L"DedicatedDisplays", L"\0", g_ini_path.c_str());
    idx = 0;
    for (const auto& d : g_dedicated) {
        if (d.checked) {
            auto key = L"DisplayId" + std::to_wstring(idx++);
            WritePrivateProfileStringW(L"DedicatedDisplays", key.c_str(),
                                       std::to_wstring(d.display_id).c_str(), g_ini_path.c_str());
        }
    }
    WritePrivateProfileStringW(L"DedicatedDisplays", L"Count",
                               std::to_wstring(idx).c_str(), g_ini_path.c_str());
}

static void load_settings()
{
    g_enabled = GetPrivateProfileIntW(L"General", L"Enabled", 1, g_ini_path.c_str()) != 0;

    int count = GetPrivateProfileIntW(L"Monitors", L"Count", 0, g_ini_path.c_str());
    std::vector<std::wstring> saved;
    for (int i = 0; i < count; ++i) {
        wchar_t buf[256] = {};
        auto key = L"Output" + std::to_wstring(i);
        GetPrivateProfileStringW(L"Monitors", key.c_str(), L"", buf, 256, g_ini_path.c_str());
        if (buf[0]) saved.emplace_back(buf);
    }

    for (auto& m : g_monitors) {
        m.checked = false;
        for (const auto& s : saved) {
            if (m.device_name == s) { m.checked = true; break; }
        }
    }

    // Load dedicated display selections
    int dd_count = GetPrivateProfileIntW(L"DedicatedDisplays", L"Count", 0, g_ini_path.c_str());
    std::vector<uint32_t> saved_ids;
    for (int i = 0; i < dd_count; ++i) {
        wchar_t buf[64] = {};
        auto key = L"DisplayId" + std::to_wstring(i);
        GetPrivateProfileStringW(L"DedicatedDisplays", key.c_str(), L"0", buf, 64, g_ini_path.c_str());
        uint32_t id = static_cast<uint32_t>(wcstoul(buf, nullptr, 10));
        if (id != 0) saved_ids.push_back(id);
    }

    for (auto& d : g_dedicated) {
        d.checked = false;
        for (auto id : saved_ids) {
            if (d.display_id == id) { d.checked = true; break; }
        }
    }
}

// ─── Windows startup registry ───────────────────────────────────────────

static bool is_autostart_enabled()
{
    HKEY key;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, REGISTRY_KEY, 0, KEY_READ, &key) != ERROR_SUCCESS)
        return false;
    DWORD size = 0;
    bool found = (RegQueryValueExW(key, REGISTRY_VAL, nullptr, nullptr, nullptr, &size) == ERROR_SUCCESS);
    RegCloseKey(key);
    return found;
}

static void set_autostart(bool enable)
{
    HKEY key;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, REGISTRY_KEY, 0, KEY_WRITE, &key) != ERROR_SUCCESS)
        return;
    if (enable) {
        std::wstring cmd = L"\"" + g_exe_path + L"\" --autostart";
        RegSetValueExW(key, REGISTRY_VAL, 0, REG_SZ,
                       reinterpret_cast<const BYTE*>(cmd.c_str()),
                       static_cast<DWORD>((cmd.size() + 1) * sizeof(wchar_t)));
    } else {
        RegDeleteValueW(key, REGISTRY_VAL);
    }
    RegCloseKey(key);
    g_autostart = enable;
}

// ─── Monitor enumeration ────────────────────────────────────────────────

static void enumerate_monitors()
{
    auto old = std::move(g_monitors);
    g_monitors.clear();

    DISPLAY_DEVICEW dd{};
    dd.cb = sizeof(dd);
    for (DWORD i = 0; EnumDisplayDevicesW(nullptr, i, &dd, 0); ++i) {
        if (!(dd.StateFlags & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP)) {
            dd.cb = sizeof(dd);
            continue;
        }

        monitor_info mi{};
        mi.device_name  = dd.DeviceName;
        mi.adapter_name = dd.DeviceString;
        mi.is_primary   = (dd.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE) != 0;

        DISPLAY_DEVICEW mon{};
        mon.cb = sizeof(mon);
        if (EnumDisplayDevicesW(dd.DeviceName, 0, &mon, 0)) {
            mi.monitor_name = mon.DeviceString;
            mi.monitor_id   = mon.DeviceID;
        }

        DEVMODEW dm{};
        dm.dmSize = sizeof(dm);
        if (EnumDisplaySettingsW(dd.DeviceName, ENUM_CURRENT_SETTINGS, &dm)) {
            mi.rect.left   = dm.dmPosition.x;
            mi.rect.top    = dm.dmPosition.y;
            mi.rect.right  = dm.dmPosition.x + static_cast<LONG>(dm.dmPelsWidth);
            mi.rect.bottom = dm.dmPosition.y + static_cast<LONG>(dm.dmPelsHeight);
        }

        mi.checked = false;
        for (const auto& o : old) {
            if (o.device_name == mi.device_name) { mi.checked = o.checked; break; }
        }

        g_monitors.push_back(std::move(mi));
        dd.cb = sizeof(dd);
    }
}

// ─── Blanker windows ────────────────────────────────────────────────────

static LRESULT CALLBACK blanker_wnd_proc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
        case WM_ERASEBKGND: {
            RECT rc;
            GetClientRect(hwnd, &rc);
            FillRect(reinterpret_cast<HDC>(wParam), &rc, g_black_brush);
            return 1;
        }
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            FillRect(hdc, &ps.rcPaint, g_black_brush);
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_CLOSE:        return 0;
        case WM_MOUSEACTIVATE: return MA_NOACTIVATEANDEAT;
        case WM_SETCURSOR:    SetCursor(nullptr); return TRUE;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static void destroy_blanker_windows()
{
    for (HWND h : g_blanker_windows)
        DestroyWindow(h);
    g_blanker_windows.clear();
}

static void create_blanker_windows()
{
    destroy_blanker_windows();
    if (!g_enabled) return;

    for (const auto& m : g_monitors) {
        if (!m.checked) continue;
        int w = m.rect.right - m.rect.left;
        int h = m.rect.bottom - m.rect.top;
        if (w <= 0 || h <= 0) continue;

        HWND hwnd = CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
            L"CasparDisplayBlanker", L"",
            WS_POPUP | WS_VISIBLE | WS_DISABLED,
            m.rect.left, m.rect.top, w, h,
            nullptr, nullptr, g_instance, nullptr);
        if (hwnd) {
            ShowWindow(hwnd, SW_SHOWNOACTIVATE);
            g_blanker_windows.push_back(hwnd);
        }
    }
}

static void update_tray_tooltip()
{
    std::wstring tip = L"CasparCG Display Blanker";
    if (g_enabled) {
        int dd_count = 0;
        for (const auto& d : g_dedicated)
            if (d.we_own) dd_count++;

        tip += L" \x2014 " + std::to_wstring(g_blanker_windows.size()) + L" blanked";
        if (dd_count > 0)
            tip += L", " + std::to_wstring(dd_count) + L" dedicated";
    } else {
        tip += L" \x2014 DISABLED";
    }
    wcsncpy_s(g_nid.szTip, tip.c_str(), _TRUNCATE);
    Shell_NotifyIconW(NIM_MODIFY, &g_nid);
}

// ─── Configuration window ───────────────────────────────────────────────

static const int P  = 12;   // padding
static const int RH = 22;   // row height
static const int BW = 94;   // button width
static const int BH = 28;   // button height
static const int WW = 480;  // window width (client)

static LRESULT CALLBACK config_wnd_proc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
        case WM_CREATE: {
            enumerate_monitors();
            if (g_nvapi_loaded)
                enumerate_dedicated_displays();
            load_settings();

            HFONT font = static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
            int cw = WW - 2 * P; // content width
            int y  = P;

            // Instruction label
            auto mk = [&](const wchar_t* cls, const wchar_t* text, DWORD style,
                          int x, int cy, UINT id = 0) -> HWND {
                HWND h = CreateWindowExW(0, cls, text, WS_CHILD | WS_VISIBLE | style,
                    x, y, cw - (x - P), cy, hwnd,
                    reinterpret_cast<HMENU>(static_cast<UINT_PTR>(id)), g_instance, nullptr);
                SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
                return h;
            };

            mk(L"STATIC",
               L"Select display outputs to keep black when CasparCG is not rendering:",
               SS_LEFT, P, RH * 2);
            y += RH * 2 + 4;

            // Separator
            CreateWindowExW(0, L"STATIC", nullptr, WS_CHILD | WS_VISIBLE | SS_ETCHEDHORZ,
                P, y, cw, 2, hwnd, nullptr, g_instance, nullptr);
            y += 8;

            // Monitor checkboxes (desktop displays — blanked via black windows)
            for (size_t i = 0; i < g_monitors.size(); ++i) {
                const auto& m = g_monitors[i];
                int mw = m.rect.right - m.rect.left;
                int mh = m.rect.bottom - m.rect.top;

                std::wstring text;
                if (m.is_primary) text += L"\x2605 "; // ★ star prefix for primary

                std::wstring devname = m.device_name;
                if (devname.size() > 4 && devname.substr(0, 4) == L"\\\\.\\")
                    devname = devname.substr(4);
                text += devname;

                text += L"  \x2014  ";
                if (!m.monitor_name.empty())     text += m.monitor_name;
                else if (!m.adapter_name.empty()) text += m.adapter_name;
                else                              text += L"(unknown)";
                text += L"  (" + std::to_wstring(mw) + L"\x00D7" + std::to_wstring(mh) + L")";
                if (m.is_primary) text += L"  [Windows main display]";

                HWND chk = mk(L"BUTTON", text.c_str(), BS_AUTOCHECKBOX,
                              P + 8, RH, static_cast<UINT>(IDC_MONITOR_BASE + i));
                if (m.checked)
                    SendMessageW(chk, BM_SETCHECK, BST_CHECKED, 0);
                y += RH + 2;
            }

            // ─── Dedicated displays section (NvAPI, Pro GPUs only) ──────────
            if (!g_dedicated.empty()) {
                y += 8;
                CreateWindowExW(0, L"STATIC", nullptr, WS_CHILD | WS_VISIBLE | SS_ETCHEDHORZ,
                    P, y, cw, 2, hwnd, nullptr, g_instance, nullptr);
                y += 8;

                mk(L"STATIC",
                   L"Dedicated GPU outputs (driver-level blanking \x2014 survives crashes):",
                   SS_LEFT, P, RH, IDC_DEDICATED_LABEL);
                y += RH + 4;

                for (size_t i = 0; i < g_dedicated.size(); ++i) {
                    const auto& d = g_dedicated[i];

                    std::wstring text = L"Display ID " + std::to_wstring(d.display_id);
                    if (d.is_mosaic) text += L"  [Mosaic]";
                    if (d.is_acquired && !d.we_own) text += L"  [in use by another app]";

                    DWORD style = BS_AUTOCHECKBOX;
                    HWND chk = mk(L"BUTTON", text.c_str(), style,
                                  P + 8, RH, static_cast<UINT>(IDC_DEDICATED_BASE + i));
                    if (d.checked || d.we_own)
                        SendMessageW(chk, BM_SETCHECK, BST_CHECKED, 0);
                    if (d.is_acquired && !d.we_own)
                        EnableWindow(chk, FALSE); // Can't acquire — another process has it

                    y += RH + 2;
                }
            } else if (g_nvapi_loaded) {
                y += 8;
                CreateWindowExW(0, L"STATIC", nullptr, WS_CHILD | WS_VISIBLE | SS_ETCHEDHORZ,
                    P, y, cw, 2, hwnd, nullptr, g_instance, nullptr);
                y += 8;

                mk(L"STATIC",
                   L"No dedicated displays found. Configure in NVIDIA Control Panel.",
                   SS_LEFT, P, RH);
                y += RH + 2;
            }

            y += 8;
            CreateWindowExW(0, L"STATIC", nullptr, WS_CHILD | WS_VISIBLE | SS_ETCHEDHORZ,
                P, y, cw, 2, hwnd, nullptr, g_instance, nullptr);
            y += 10;

            // Enable blanking
            {
                HWND h = mk(L"BUTTON", L"Enable blanking", BS_AUTOCHECKBOX,
                             P + 8, RH, IDC_ENABLE_CHK);
                if (g_enabled) SendMessageW(h, BM_SETCHECK, BST_CHECKED, 0);
                y += RH + 4;
            }

            // Start with Windows
            {
                HWND h = mk(L"BUTTON", L"Start with Windows", BS_AUTOCHECKBOX,
                             P + 8, RH, IDC_AUTOSTART_CHK);
                if (is_autostart_enabled()) SendMessageW(h, BM_SETCHECK, BST_CHECKED, 0);
                y += RH + 16;
            }

            // Buttons — right-aligned
            {
                int bx = WW - P - BW;
                HWND cb = CreateWindowExW(0, L"BUTTON", L"Close",
                    WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                    bx, y, BW, BH, hwnd,
                    reinterpret_cast<HMENU>(static_cast<UINT_PTR>(IDC_CLOSE_BTN)),
                    g_instance, nullptr);
                SendMessageW(cb, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);

                bx -= BW + 12;
                HWND sb = CreateWindowExW(0, L"BUTTON", L"Save && Apply",
                    WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
                    bx, y, BW + 12, BH, hwnd,
                    reinterpret_cast<HMENU>(static_cast<UINT_PTR>(IDC_SAVE_BTN)),
                    g_instance, nullptr);
                SendMessageW(sb, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
            }
            y += BH + P;

            // Size window to fit
            RECT wr = {0, 0, WW, y};
            DWORD style = GetWindowLongW(hwnd, GWL_STYLE);
            AdjustWindowRectEx(&wr, style, FALSE, 0);
            int ww = wr.right - wr.left;
            int wh = wr.bottom - wr.top;
            int sx = GetSystemMetrics(SM_CXSCREEN);
            int sy = GetSystemMetrics(SM_CYSCREEN);
            SetWindowPos(hwnd, nullptr, (sx - ww) / 2, (sy - wh) / 2, ww, wh, SWP_NOZORDER);
            return 0;
        }

        case WM_COMMAND:
            switch (LOWORD(wParam)) {
                case IDC_SAVE_BTN: {
                    // Snapshot current state before applying changes
                    auto snapshot = take_snapshot();

                    // Read new state from UI
                    for (size_t i = 0; i < g_monitors.size(); ++i) {
                        HWND chk = GetDlgItem(hwnd, static_cast<int>(IDC_MONITOR_BASE + i));
                        if (chk)
                            g_monitors[i].checked = (SendMessageW(chk, BM_GETCHECK, 0, 0) == BST_CHECKED);
                    }

                    // Read dedicated display checkboxes
                    for (size_t i = 0; i < g_dedicated.size(); ++i) {
                        HWND chk = GetDlgItem(hwnd, static_cast<int>(IDC_DEDICATED_BASE + i));
                        if (chk)
                            g_dedicated[i].checked = (SendMessageW(chk, BM_GETCHECK, 0, 0) == BST_CHECKED);
                    }

                    HWND en = GetDlgItem(hwnd, IDC_ENABLE_CHK);
                    if (en) g_enabled = (SendMessageW(en, BM_GETCHECK, 0, 0) == BST_CHECKED);

                    HWND as = GetDlgItem(hwnd, IDC_AUTOSTART_CHK);
                    bool want_as = as && (SendMessageW(as, BM_GETCHECK, 0, 0) == BST_CHECKED);
                    if (want_as != g_autostart) set_autostart(want_as);

                    // Apply changes immediately (so user can see the effect)
                    if (g_enabled) {
                        create_blanker_windows();
                        acquire_dedicated_displays();
                    } else {
                        destroy_blanker_windows();
                        release_all_dedicated_displays();
                    }
                    update_tray_tooltip();

                    // Close config window before showing confirmation
                    DestroyWindow(hwnd);

                    // Show confirmation countdown on primary monitor
                    if (show_confirmation_countdown()) {
                        // User confirmed — save to disk
                        save_settings();
                    } else {
                        // Timed out or user clicked Revert — restore previous state
                        restore_snapshot(snapshot);
                        if (g_enabled) {
                            create_blanker_windows();
                            acquire_dedicated_displays();
                        } else {
                            destroy_blanker_windows();
                            release_all_dedicated_displays();
                        }
                        update_tray_tooltip();
                    }
                    break;
                }
                case IDC_CLOSE_BTN:
                    DestroyWindow(hwnd);
                    break;
            }
            return 0;

        case WM_CLOSE:
            DestroyWindow(hwnd);
            return 0;

        case WM_DESTROY:
            g_config_hwnd = nullptr;
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static void show_config_window()
{
    if (g_config_hwnd) {
        SetForegroundWindow(g_config_hwnd);
        return;
    }

    g_config_hwnd = CreateWindowExW(
        0, L"CasparBlankerCfg",
        L"CasparCG Display Blanker",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, WW, 400,
        nullptr, nullptr, g_instance, nullptr);

    if (g_config_hwnd)
        SetForegroundWindow(g_config_hwnd);
}

// ─── Confirmation countdown dialog ─────────────────────────────────────
//
// Like Windows display settings: shows a countdown on the primary monitor.
// If the user doesn't confirm, changes are reverted. This prevents
// accidentally blanking the control monitor and locking yourself out.

static const int CONFIRM_SECONDS = 15;
static const UINT_PTR IDT_COUNTDOWN = 9001;
static int g_countdown_remaining = 0;
static HWND g_confirm_hwnd = nullptr;
static bool g_confirm_result = false; // true = keep, false = revert
static const UINT IDC_CONFIRM_LABEL = 9010;
static const UINT IDC_CONFIRM_YES   = 9011;
static const UINT IDC_CONFIRM_NO    = 9012;

static void update_confirm_label()
{
    if (!g_confirm_hwnd) return;
    HWND label = GetDlgItem(g_confirm_hwnd, IDC_CONFIRM_LABEL);
    if (!label) return;

    std::wstring text = L"Do you want to keep these display settings?\n\n"
                        L"Reverting in " + std::to_wstring(g_countdown_remaining) + L" seconds...";
    SetWindowTextW(label, text.c_str());
}

static LRESULT CALLBACK confirm_wnd_proc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
        case WM_CREATE: {
            HFONT font = static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));

            // Message text
            HWND lbl = CreateWindowExW(0, L"STATIC",
                L"Do you want to keep these display settings?\n\nReverting in 15 seconds...",
                WS_CHILD | WS_VISIBLE | SS_CENTER,
                20, 20, 320, 60, hwnd,
                reinterpret_cast<HMENU>(static_cast<UINT_PTR>(IDC_CONFIRM_LABEL)),
                g_instance, nullptr);
            SendMessageW(lbl, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);

            // "Keep Changes" button
            HWND yes = CreateWindowExW(0, L"BUTTON", L"Keep Changes",
                WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
                40, 92, 130, 32, hwnd,
                reinterpret_cast<HMENU>(static_cast<UINT_PTR>(IDC_CONFIRM_YES)),
                g_instance, nullptr);
            SendMessageW(yes, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);

            // "Revert" button
            HWND no = CreateWindowExW(0, L"BUTTON", L"Revert",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                190, 92, 130, 32, hwnd,
                reinterpret_cast<HMENU>(static_cast<UINT_PTR>(IDC_CONFIRM_NO)),
                g_instance, nullptr);
            SendMessageW(no, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);

            // Start countdown timer (1 second intervals)
            g_countdown_remaining = CONFIRM_SECONDS;
            SetTimer(hwnd, IDT_COUNTDOWN, 1000, nullptr);
            return 0;
        }

        case WM_TIMER:
            if (wParam == IDT_COUNTDOWN) {
                g_countdown_remaining--;
                if (g_countdown_remaining <= 0) {
                    KillTimer(hwnd, IDT_COUNTDOWN);
                    g_confirm_result = false;
                    DestroyWindow(hwnd);
                } else {
                    update_confirm_label();
                }
            }
            return 0;

        case WM_COMMAND:
            switch (LOWORD(wParam)) {
                case IDC_CONFIRM_YES:
                    KillTimer(hwnd, IDT_COUNTDOWN);
                    g_confirm_result = true;
                    DestroyWindow(hwnd);
                    break;
                case IDC_CONFIRM_NO:
                    KillTimer(hwnd, IDT_COUNTDOWN);
                    g_confirm_result = false;
                    DestroyWindow(hwnd);
                    break;
            }
            return 0;

        case WM_CLOSE:
            KillTimer(hwnd, IDT_COUNTDOWN);
            g_confirm_result = false;
            DestroyWindow(hwnd);
            return 0;

        case WM_DESTROY:
            g_confirm_hwnd = nullptr;
            PostMessageW(g_tray_hwnd, WM_APP + 99, 0, 0); // Signal main loop to continue
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// Shows the confirmation countdown on the primary monitor.
// Returns true if user confirmed, false if timed out or reverted.
static bool show_confirmation_countdown()
{
    // Register a class for the confirmation window
    static bool registered = false;
    if (!registered) {
        WNDCLASSEXW wc{};
        wc.cbSize        = sizeof(wc);
        wc.lpfnWndProc   = confirm_wnd_proc;
        wc.hInstance      = g_instance;
        wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
        wc.hCursor       = LoadCursorW(nullptr, IDC_ARROW);
        wc.hIcon         = g_app_icon;
        wc.lpszClassName = L"CasparBlankerConfirm";
        RegisterClassExW(&wc);
        registered = true;
    }

    // Find the primary monitor to position the dialog there
    RECT primary_rect = {0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)};
    for (const auto& m : g_monitors) {
        if (m.is_primary) { primary_rect = m.rect; break; }
    }

    int dw = 360, dh = 150;
    RECT wr = {0, 0, dw, dh};
    AdjustWindowRectEx(&wr, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, FALSE, WS_EX_TOPMOST);
    int ww = wr.right - wr.left;
    int wh = wr.bottom - wr.top;
    int dx = primary_rect.left + ((primary_rect.right - primary_rect.left) - ww) / 2;
    int dy = primary_rect.top + ((primary_rect.bottom - primary_rect.top) - wh) / 2;

    g_confirm_result = false;
    g_confirm_hwnd = CreateWindowExW(
        WS_EX_TOPMOST,
        L"CasparBlankerConfirm",
        L"Confirm Display Settings",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_VISIBLE,
        dx, dy, ww, wh,
        nullptr, nullptr, g_instance, nullptr);

    if (!g_confirm_hwnd)
        return true; // Can't show dialog — assume confirmed

    SetForegroundWindow(g_confirm_hwnd);

    // Run a nested message loop until the confirmation window is closed
    MSG msg;
    while (g_confirm_hwnd && GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    return g_confirm_result;
}

// ─── Tray icon ──────────────────────────────────────────────────────────

static LRESULT CALLBACK tray_wnd_proc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
        case WM_TRAYICON:
            if (LOWORD(lParam) == WM_RBUTTONUP) {
                POINT pt;
                GetCursorPos(&pt);
                SetForegroundWindow(hwnd);

                HMENU menu = CreatePopupMenu();
                AppendMenuW(menu, MF_STRING, IDM_CONFIGURE, L"Configure...");
                AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
                AppendMenuW(menu, MF_STRING, IDM_TOGGLE,
                            g_enabled ? L"Disable Blanking" : L"Enable Blanking");
                AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
                AppendMenuW(menu, MF_STRING, IDM_EXIT, L"Exit");

                TrackPopupMenu(menu, TPM_RIGHTBUTTON, pt.x, pt.y, 0, hwnd, nullptr);
                DestroyMenu(menu);
                PostMessageW(hwnd, WM_NULL, 0, 0);
            } else if (LOWORD(lParam) == WM_LBUTTONDBLCLK) {
                show_config_window();
            }
            return 0;

        case WM_COMMAND:
            switch (LOWORD(wParam)) {
                case IDM_CONFIGURE:
                    show_config_window();
                    break;
                case IDM_TOGGLE:
                    g_enabled = !g_enabled;
                    save_settings();
                    if (g_enabled) {
                        create_blanker_windows();
                        acquire_dedicated_displays();
                    } else {
                        destroy_blanker_windows();
                        release_all_dedicated_displays();
                    }
                    update_tray_tooltip();
                    break;
                case IDM_EXIT:
                    PostQuitMessage(0);
                    break;
            }
            return 0;

        case WM_DESTROY:
            Shell_NotifyIconW(NIM_DELETE, &g_nid);
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static void create_tray_icon()
{
    WNDCLASSEXW wc{};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = tray_wnd_proc;
    wc.hInstance      = g_instance;
    wc.lpszClassName = L"CasparBlankerTray";
    RegisterClassExW(&wc);

    g_tray_hwnd = CreateWindowExW(0, L"CasparBlankerTray", L"", 0,
                                  0, 0, 0, 0, HWND_MESSAGE, nullptr, g_instance, nullptr);

    g_nid.cbSize           = sizeof(g_nid);
    g_nid.hWnd             = g_tray_hwnd;
    g_nid.uID              = 1;
    g_nid.uFlags           = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    g_nid.uCallbackMessage = WM_TRAYICON;
    g_nid.hIcon            = g_app_icon;
    wcscpy_s(g_nid.szTip, L"CasparCG Display Blanker");

    Shell_NotifyIconW(NIM_ADD, &g_nid);
}

// ─── CLI helpers ────────────────────────────────────────────────────────

static void apply_cli_match(const std::vector<std::wstring>& patterns)
{
    for (auto& m : g_monitors) {
        m.checked = false;
        for (const auto& p : patterns) {
            if (icontains(m.monitor_name, p) || icontains(m.monitor_id, p) ||
                icontains(m.adapter_name, p) || icontains(m.device_name, p)) {
                m.checked = true;
                break;
            }
        }
    }
}

// ─── WinMain (no console window) ────────────────────────────────────────

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR lpCmdLine, int)
{
    g_instance  = hInstance;
    g_exe_path  = [] { wchar_t b[MAX_PATH]={}; GetModuleFileNameW(nullptr,b,MAX_PATH); return std::wstring(b); }();
    g_ini_path  = get_exe_directory() + L"display_blanker.ini";
    g_autostart = is_autostart_enabled();

    // Load app icon from resource, fall back to system icon
    g_app_icon = LoadIconW(hInstance, MAKEINTRESOURCEW(IDI_APP_ICON));
    if (!g_app_icon)
        g_app_icon = LoadIconW(nullptr, IDI_APPLICATION);

    // Parse command line
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(lpCmdLine, &argc);

    bool cli_autostart = false;
    bool cli_all       = false;
    std::vector<std::wstring> cli_match;

    if (argv) {
        for (int i = 0; i < argc; ++i) {
            if (wcscmp(argv[i], L"--autostart") == 0)
                cli_autostart = true;
            else if (wcscmp(argv[i], L"--match") == 0 && i + 1 < argc)
                cli_match.push_back(argv[++i]);
            else if (wcscmp(argv[i], L"--all") == 0)
                cli_all = true;
        }
        LocalFree(argv);
    }

    // Single instance
    g_mutex = CreateMutexW(nullptr, FALSE, MUTEX_NAME);
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        if (g_mutex) CloseHandle(g_mutex);
        return 0;
    }

    // Init common controls
    INITCOMMONCONTROLSEX icc{};
    icc.dwSize = sizeof(icc);
    icc.dwICC  = ICC_STANDARD_CLASSES;
    InitCommonControlsEx(&icc);

    g_black_brush = CreateSolidBrush(RGB(0, 0, 0));

    // Register window classes
    {
        WNDCLASSEXW wc{};
        wc.cbSize        = sizeof(wc);
        wc.style         = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc   = blanker_wnd_proc;
        wc.hInstance      = g_instance;
        wc.hbrBackground = g_black_brush;
        wc.lpszClassName = L"CasparDisplayBlanker";
        RegisterClassExW(&wc);
    }
    {
        WNDCLASSEXW wc{};
        wc.cbSize        = sizeof(wc);
        wc.lpfnWndProc   = config_wnd_proc;
        wc.hInstance      = g_instance;
        wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
        wc.hCursor       = LoadCursorW(nullptr, IDC_ARROW);
        wc.hIcon         = g_app_icon;
        wc.lpszClassName = L"CasparBlankerCfg";
        RegisterClassExW(&wc);
    }

    // Enumerate monitors
    enumerate_monitors();

    // Try to load NvAPI for dedicated display support
    load_nvapi();
    if (g_nvapi_loaded)
        enumerate_dedicated_displays();

    // Determine initial behavior
    bool show_config = false;

    if (!cli_match.empty()) {
        apply_cli_match(cli_match);
        g_enabled = true;
        save_settings();
    } else if (cli_all) {
        for (auto& m : g_monitors) m.checked = !m.is_primary;
        g_enabled = true;
        save_settings();
    } else if (cli_autostart) {
        load_settings();
    } else {
        load_settings();
        show_config = true;
    }

    // Create tray icon
    create_tray_icon();

    // Create blanker windows
    if (g_enabled)
        create_blanker_windows();

    // Acquire dedicated displays
    if (g_enabled)
        acquire_dedicated_displays();

    update_tray_tooltip();

    // Show config on interactive launch
    if (show_config)
        show_config_window();

    // Message loop
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        if (g_config_hwnd && IsDialogMessageW(g_config_hwnd, &msg))
            continue;
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    // Cleanup
    destroy_blanker_windows();
    release_all_dedicated_displays();
    Shell_NotifyIconW(NIM_DELETE, &g_nid);
    if (g_tray_hwnd)   DestroyWindow(g_tray_hwnd);
    if (g_config_hwnd) DestroyWindow(g_config_hwnd);
    DeleteObject(g_black_brush);
    if (g_mutex) CloseHandle(g_mutex);
    if (g_nvapi_unload) g_nvapi_unload();

    return 0;
}
