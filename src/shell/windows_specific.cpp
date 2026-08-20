/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Robert Nagy, ronag89@gmail.com
 */
#include "platform_specific.h"

#include <common/env.h>
#include <common/log.h>
#include <common/os/windows/windows.h>

#include <atlbase.h>
#include <mmsystem.h>
#include <winnt.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <thread>

#include <fcntl.h>
#include <io.h>

// NOTE: This is needed in order to make CComObject work since this is not a real ATL project.
CComModule                               _AtlModule;
extern __declspec(selectany) CAtlModule* _pAtlModule = &_AtlModule;

extern "C" {
// Force discrete nVidia GPU
// (http://developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf)
_declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
// Force discrete AMD GPU (https://community.amd.com/thread/169965 /
// https://gpuopen.com/amdpowerxpressrequesthighperformance/)
_declspec(dllexport) DWORD AmdPowerXpressRequestHighPerformance = 0x00000001;
}

#include <common/os/windows/process.h>

namespace caspar {

// TEMPORARY DIAGNOSTIC (remove). A vectored handler sees the FIRST-CHANCE exception, before
// /EHa translates an access violation into a C++ exception that a `catch (...)` swallows --
// which is why the prores_vulkan fault has only ever been visible as "Decoder thread non-C++
// exception". Enabled only when CASPARVP_VEH_TRACE is set.
namespace {
std::atomic<int> g_veh_reports{0};

std::wstring module_for(void* addr)
{
    HMODULE mod = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(addr),
                            &mod) ||
        !mod) {
        return L"<no module>";
    }
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(mod, path, MAX_PATH);
    const auto base   = reinterpret_cast<std::uintptr_t>(mod);
    const auto offset = reinterpret_cast<std::uintptr_t>(addr) - base;
    std::wstringstream ss;
    const wchar_t*     name = wcsrchr(path, L'\\');
    ss << (name ? name + 1 : path) << L"+0x" << std::hex << offset;
    return ss.str();
}

LONG WINAPI veh_trace(EXCEPTION_POINTERS* info)
{
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION)
        return EXCEPTION_CONTINUE_SEARCH;
    // tbbmalloc probes memory and HANDLES its own access violations; they are normal and
    // they arrive in the hundreds, so they would eat any report budget.
    const auto where = module_for(info->ExceptionRecord->ExceptionAddress);
    if (where.find(L"tbbmalloc") != std::wstring::npos)
        return EXCEPTION_CONTINUE_SEARCH;
    if (g_veh_reports.fetch_add(1) >= 12)
        return EXCEPTION_CONTINUE_SEARCH;

    try {
        auto* rec = info->ExceptionRecord;
        auto* ctx = info->ContextRecord;

        std::wstringstream ss;
        ss << L"[VEH] access violation at " << rec->ExceptionAddress << L" (" << where << L")";
        ss << L" op=" << rec->ExceptionInformation[0] << L" addr=0x" << std::hex
           << rec->ExceptionInformation[1] << std::dec;

        // A jump to a null function pointer leaves RIP at 0 and the RETURN ADDRESS on top of
        // the stack, so [RSP] names the caller -- the only way to find out whose pointer it
        // was. For any other fault RSP is not a return address, hence the guard.
        if (reinterpret_cast<std::uintptr_t>(rec->ExceptionAddress) == 0 && ctx->Rsp) {
            void* ret = nullptr;
            if (ReadProcessMemory(GetCurrentProcess(),
                                  reinterpret_cast<void*>(ctx->Rsp),
                                  &ret,
                                  sizeof(ret),
                                  nullptr) &&
                ret) {
                ss << L"\n[VEH] called from " << ret << L" (" << module_for(ret) << L")";
            }
            ss << L"\n[VEH] rax=" << reinterpret_cast<void*>(ctx->Rax) << L" rbx="
               << reinterpret_cast<void*>(ctx->Rbx) << L" rcx=" << reinterpret_cast<void*>(ctx->Rcx)
               << L" rdx=" << reinterpret_cast<void*>(ctx->Rdx);

            // Walk a few more stack slots: the immediate caller may itself be a thunk.
            for (int i = 1; i <= 8; ++i) {
                void* slot = nullptr;
                if (!ReadProcessMemory(GetCurrentProcess(),
                                       reinterpret_cast<void*>(ctx->Rsp + i * sizeof(void*)),
                                       &slot,
                                       sizeof(slot),
                                       nullptr))
                    break;
                if (!slot)
                    continue;
                const auto m = module_for(slot);
                if (m != L"<no module>")
                    ss << L"\n[VEH]   stack[" << i << L"] " << slot << L" (" << m << L")";
            }
        }

        CASPAR_LOG(error) << ss.str();
    } catch (...) {
    }

    return EXCEPTION_CONTINUE_SEARCH;
}
} // namespace

LONG WINAPI UserUnhandledExceptionFilter(EXCEPTION_POINTERS* info)
{
    try {
        CASPAR_LOG(fatal) << L"#######################\n UNHANDLED EXCEPTION: \n"
                          << L"Address:" << info->ExceptionRecord->ExceptionAddress << L"\n"
                          << L"Code:" << info->ExceptionRecord->ExceptionCode << L"\n"
                          << L"Flag:" << info->ExceptionRecord->ExceptionFlags << L"\n"
                          << L"Info:" << (unsigned __int64)info->ExceptionRecord->ExceptionInformation << L"\n"
                          << L"Continuing execution. \n#######################";

        CASPAR_LOG_CURRENT_CALL_STACK();
    } catch (...) {
    }

    return EXCEPTION_EXECUTE_HANDLER;
}

void setup_process_scheduling()
{
    // Increase time precision. This will increase accuracy of function like Sleep(1) from 10 ms to 1 ms.
    static struct inc_prec
    {
        inc_prec() { timeBeginPeriod(1); }
        ~inc_prec() { timeEndPeriod(1); }
    } inc_prec;

    // Stop the OS taking back the resolution requested above when no window is visible. CEF paces
    // audio off a timer rather than a device clock, so its subprocesses need this too.
    disable_process_power_throttling();
}

void setup_prerequisites()
{
    // Enable utf8 console input and output
    _setmode(_fileno(stdout), _O_U8TEXT);
    _setmode(_fileno(stdin), _O_U16TEXT);

    SetUnhandledExceptionFilter(UserUnhandledExceptionFilter);

    if (std::getenv("CASPARVP_VEH_TRACE"))
        AddVectoredExceptionHandler(1, veh_trace);
}

void change_icon(const HICON hNewIcon)
{
    auto hMod              = ::LoadLibrary(L"Kernel32.dll");
    using SCI              = DWORD(__stdcall*)(HICON);
    auto pfnSetConsoleIcon = reinterpret_cast<SCI>(::GetProcAddress(hMod, "SetConsoleIcon"));
    pfnSetConsoleIcon(hNewIcon);
    ::FreeLibrary(hMod);
}

void setup_console_window()
{
    auto  hOut           = GetStdHandle(STD_OUTPUT_HANDLE);
    auto  hIn            = GetStdHandle(STD_INPUT_HANDLE);
    DWORD dwPreviousMode = 0;

    if (hIn != INVALID_HANDLE_VALUE && GetConsoleMode(hIn, &dwPreviousMode)) {
        dwPreviousMode &= ~ENABLE_QUICK_EDIT_MODE | ENABLE_EXTENDED_FLAGS; // disable quick edit mode
        dwPreviousMode &= ENABLE_PROCESSED_INPUT | ~ENABLE_MOUSE_INPUT;    // allow mouse wheel scrolling
        SetConsoleMode(hIn, dwPreviousMode);
    }

    // Disable close button in console to avoid shutdown without cleanup.
    EnableMenuItem(GetSystemMenu(GetConsoleWindow(), FALSE), SC_CLOSE, MF_GRAYED);
    DrawMenuBar(GetConsoleWindow());
    SetConsoleCtrlHandler(nullptr, true);

    if (hOut != INVALID_HANDLE_VALUE) {
        // Configure console size and position.
        auto coord = GetLargestConsoleWindowSize(hOut);
        coord.X /= 2;
        coord.Y *= 10;
        SetConsoleScreenBufferSize(hOut, coord);

        SMALL_RECT DisplayArea = {0, 0, 0, 0};
        DisplayArea.Right      = coord.X - 1;
        DisplayArea.Bottom     = (coord.Y / 10 - 1) / 2;
        SetConsoleWindowInfo(hOut, TRUE, &DisplayArea);
    }

    change_icon(::LoadIcon(GetModuleHandle(nullptr), MAKEINTRESOURCE(101)));

    // Set console title.
    std::wstringstream str;
    str << "CasparCG Server " << env::version() << L" x64 ";
#ifdef COMPILE_RELEASE
    str << " Release";
#elif COMPILE_PROFILE
    str << " Profile";
#elif COMPILE_DEVELOP
    str << " Develop";
#elif COMPILE_DEBUG
    str << " Debug";
#endif
    SetConsoleTitle(str.str().c_str());
}

void increase_process_priority() { SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS); }

void wait_for_keypress()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::system("pause");
}

std::shared_ptr<void> setup_debugging_environment()
{
#ifdef _DEBUG
    HANDLE hLogFile = CreateFile(
        L"crt_log.txt", GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    std::shared_ptr<void> crt_log(nullptr, [](HANDLE h) { ::CloseHandle(h); });

    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, hLogFile);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, hLogFile);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, hLogFile);

    return crt_log;
#else
    return nullptr;
#endif
}

void wait_for_remote_debugging()
{
#ifdef _DEBUG
    MessageBox(nullptr, L"Now is the time to connect for remote debugging...", L"Debug", MB_OK | MB_TOPMOST);
#endif
}

} // namespace caspar
