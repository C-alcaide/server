/*
 * Copyright 2013 Sveriges Television AB http://casparcg.com/
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

#include "html.h"
#include "util.h"

#include "producer/html_cg_proxy.h"
#include "producer/html_producer.h"

#include <common/env.h>
#include <common/executor.h>
#include <common/future.h>

#include <core/producer/cg_proxy.h>

#include <boost/asio.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/range/algorithm/remove_if.hpp>

#include <memory>
#include <utility>

#ifdef __APPLE__
#include <boost/dll/runtime_symbol_info.hpp>
#include <pthread.h>
#include <sched.h>
#endif

#pragma warning(push)
#pragma warning(disable : 4458)
#include <include/cef_app.h>
#include <include/cef_version.h>
#pragma warning(pop)

#ifdef WIN32
#include <accelerator/d3d/d3d_device.h>
#endif

namespace caspar::html {

std::unique_ptr<executor> g_cef_executor;

void caspar_log(const CefRefPtr<CefBrowser>&        browser,
                boost::log::trivial::severity_level level,
                const std::string&                  message)
{
    if (browser != nullptr) {
        auto msg = CefProcessMessage::Create(LOG_MESSAGE_NAME);
        msg->GetArgumentList()->SetInt(0, level);
        msg->GetArgumentList()->SetString(1, message);

        CefRefPtr<CefFrame> mainFrame = browser->GetMainFrame();
        if (mainFrame) {
            mainFrame->SendProcessMessage(PID_BROWSER, msg);
        }
    }
}

class remove_handler : public CefV8Handler
{
    CefRefPtr<CefBrowser> browser_;

  public:
    explicit remove_handler(const CefRefPtr<CefBrowser>& browser)
        : browser_(browser)
    {
    }

    bool Execute(const CefString&       name,
                 CefRefPtr<CefV8Value>  object,
                 const CefV8ValueList&  arguments,
                 CefRefPtr<CefV8Value>& retval,
                 CefString&             exception) override
    {
        if (!CefCurrentlyOn(TID_RENDERER)) {
            return false;
        }

        CefRefPtr<CefFrame> mainFrame = browser_->GetMainFrame();
        if (mainFrame) {
            mainFrame->SendProcessMessage(PID_BROWSER, CefProcessMessage::Create(REMOVE_MESSAGE_NAME));
        }

        return true;
    }

    IMPLEMENT_REFCOUNTING(remove_handler);
};

class renderer_application
    : public CefApp
    , CefRenderProcessHandler
{
    std::vector<CefRefPtr<CefV8Context>> contexts_;
    const bool                           enable_gpu_;
    const bool                           shared_texture_;

  public:
    explicit renderer_application(const bool enable_gpu, const bool shared_texture)
        : enable_gpu_(enable_gpu)
        , shared_texture_(shared_texture)
    {
    }

    CefRefPtr<CefRenderProcessHandler> GetRenderProcessHandler() override { return this; }

    void
    OnContextCreated(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, CefRefPtr<CefV8Context> context) override
    {
        if (!frame->IsMain())
            return;

        caspar_log(
            browser, boost::log::trivial::trace, "context for frame " + frame->GetIdentifier().ToString() + " created");
        contexts_.push_back(context);

        auto window = context->GetGlobal();

        window->SetValue(
            "remove", CefV8Value::CreateFunction("remove", new remove_handler(browser)), V8_PROPERTY_ATTRIBUTE_NONE);

        CefRefPtr<CefV8Value>     ret;
        CefRefPtr<CefV8Exception> exception;
        bool                      injected = context->Eval(R"(
            window.caspar = window.casparcg = {};
		)",
                                      CefString(),
                                      1,
                                      ret,
                                      exception);

        if (!injected) {
            caspar_log(browser, boost::log::trivial::error, "Could not inject javascript animation code.");
        }
    }

    void OnContextReleased(CefRefPtr<CefBrowser>   browser,
                           CefRefPtr<CefFrame>     frame,
                           CefRefPtr<CefV8Context> context) override
    {
        if (!frame->IsMain())
            return;

        auto removed =
            boost::remove_if(contexts_, [&](const CefRefPtr<CefV8Context>& c) { return c->IsSame(context); });

        if (removed != contexts_.end()) {
            caspar_log(browser,
                       boost::log::trivial::trace,
                       "context for frame " + frame->GetIdentifier().ToString() + " released");
        } else {
            caspar_log(browser,
                       boost::log::trivial::warning,
                       "context for frame " + frame->GetIdentifier().ToString() + " released, but not found");
        }
    }

    void OnBrowserDestroyed(CefRefPtr<CefBrowser> browser) override { contexts_.clear(); }

    void OnBeforeCommandLineProcessing(const CefString& process_type, CefRefPtr<CefCommandLine> command_line) override
    {
        if (enable_gpu_) {
            command_line->AppendSwitch("enable-webgl");

            auto default_backend = L"";
#if __APPLE__
            // macOS: prefer Metal backend via ANGLE for best performance
            default_backend = L"metal";
#elif __unix__
            // If there is no X server, Chromium requires us to force it to the angle backend
            if (getenv("DISPLAY") == nullptr)
                default_backend = L"vulkan";
#endif

            // This gives better performance on the gpu->cpu readback, but can perform worse with intense templates
            auto backend = env::properties().get(L"configuration.html.angle-backend", default_backend);
            if (backend.size() > 0) {
                command_line->AppendSwitchWithValue("use-angle", backend);
            }
        }

#if defined(__unix__) && !defined(__APPLE__)
        // Linux: If there is no X server, use headless ozone platform
        if (getenv("DISPLAY") == nullptr) {
            command_line->AppendSwitchWithValue("ozone-platform", "headless");
        }
#endif

        command_line->AppendSwitch("disable-web-security");
        command_line->AppendSwitch("enable-begin-frame-scheduling");
        command_line->AppendSwitch("enable-media-stream");
        command_line->AppendSwitch("use-fake-ui-for-media-stream");
        command_line->AppendSwitchWithValue("autoplay-policy", "no-user-gesture-required");
        command_line->AppendSwitchWithValue("remote-allow-origins", "*");

#ifdef __APPLE__
        // macOS: Use mock keychain to prevent "Chromium Safe Storage" keychain permission dialog
        command_line->AppendSwitch("use-mock-keychain");

        // macOS: Run GPU thread in main process to avoid subprocess launch failures
        // CEF's GPU subprocess can fail to launch on macOS due to signing/sandbox issues
        command_line->AppendSwitch("in-process-gpu");
#endif

        if (process_type.empty() && !enable_gpu_) {
            // This gives more performance, but disabled gpu effects. Without it a single 1080p producer cannot be run
            // smoothly

            command_line->AppendSwitch("disable-gpu");
            command_line->AppendSwitch("disable-gpu-compositing");
            command_line->AppendSwitchWithValue("disable-gpu-vsync", "gpu");
        }
    }

    IMPLEMENT_REFCOUNTING(renderer_application);
};

bool intercept_command_line(int argc, char** argv)
{
#ifdef _WIN32
    CefMainArgs main_args;
#else
    CefMainArgs main_args(argc, argv);
#endif

    return CefExecuteProcess(main_args, CefRefPtr<CefApp>(new renderer_application(false, false)), nullptr) >= 0;
}

void init(const core::module_dependencies& dependencies)
{
    dependencies.producer_registry->register_producer_factory(L"HTML Producer", html::create_producer);

    CefMainArgs main_args;
    g_cef_executor = std::make_unique<executor>(L"cef");
    bool result    = g_cef_executor->invoke([&] {
#ifdef WIN32
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#elif defined(__APPLE__)
        // macOS: Set thread to high priority using pthread
        pthread_t          thread = pthread_self();
        struct sched_param param;
        param.sched_priority = sched_get_priority_max(SCHED_FIFO);
        pthread_setschedparam(thread, SCHED_FIFO, &param);
#endif
        const auto gpu = is_gpu_shared_texture_enabled();

        CefSettings settings;
        settings.command_line_args_disabled   = false;
        settings.no_sandbox                   = true;
        settings.remote_debugging_port        = env::properties().get(L"configuration.html.remote-debugging-port", 0);
        settings.windowless_rendering_enabled = true;

#ifdef __APPLE__
        // macOS: Configure paths for both app bundle and flat deployment
        // Get executable path and derive framework/resource locations
        auto exe_path = boost::dll::program_location();
        auto exe_dir  = exe_path.parent_path();

        // Detect if running from an app bundle (path contains .app/Contents/MacOS)
        auto exe_path_str  = exe_path.string();
        bool is_app_bundle = exe_path_str.find(".app/Contents/MacOS") != std::string::npos;

        boost::filesystem::path bundle_path;
        boost::filesystem::path data_dir;

        if (is_app_bundle) {
            // App bundle: CasparCG.app/Contents/MacOS/casparcg
            auto contents_path = exe_dir.parent_path();       // Contents
            bundle_path        = contents_path.parent_path(); // CasparCG.app
            data_dir           = contents_path / "Resources" / "data";
            CASPAR_LOG(info) << "[html] Running from app bundle: " << bundle_path.string();
        } else {
            // Flat structure: build/shell/casparcg
            bundle_path = exe_dir;
            data_dir    = exe_dir / "data";
            CASPAR_LOG(info) << "[html] Running from flat structure: " << exe_dir.string();
        }

        // Framework path is always ../Frameworks relative to executable
        auto frameworks_path = exe_dir.parent_path() / "Frameworks";

        // Framework: Contents/Frameworks/Chromium Embedded Framework.framework
        auto framework_path = frameworks_path / "Chromium Embedded Framework.framework";
        CefString(&settings.framework_dir_path).FromString(framework_path.string());

        // Resources are inside the framework bundle on macOS
        auto resources_path = framework_path / "Resources";
        CefString(&settings.resources_dir_path).FromString(resources_path.string());
        CefString(&settings.locales_dir_path).FromString(resources_path.string());

        // Set the subprocess path to the main executable (handles renderer, GPU processes)
        CefString(&settings.browser_subprocess_path).FromString(exe_path.string());

        // Set main_bundle_path appropriately for app bundle or flat structure
        if (is_app_bundle) {
            CefString(&settings.main_bundle_path).FromString(bundle_path.string());
        } else {
            CefString(&settings.main_bundle_path).FromString(exe_dir.string());
        }

        CASPAR_LOG(info) << "[html] macOS CEF paths configured:";
        CASPAR_LOG(info) << "[html]   App bundle: " << (is_app_bundle ? "yes" : "no");
        CASPAR_LOG(info) << "[html]   Framework: " << framework_path.string();
        CASPAR_LOG(info) << "[html]   Resources: " << resources_path.string();
        CASPAR_LOG(info) << "[html]   Subprocess: " << exe_path.string();
        CASPAR_LOG(info) << "[html]   Bundle path: " << bundle_path.string();
#endif

        // Set root_cache_path to prevent CEF from using shared keychain storage
        auto cache_path = env::properties().get(L"configuration.html.cache-path", L"cef-cache");
        if (!cache_path.empty()) {
            if (!boost::filesystem::path(cache_path).is_absolute()) {
                cache_path = caspar::env::initial_folder() + L"/" + cache_path;
            }
            CASPAR_LOG(info) << L"[html] Using CEF cache path: " << cache_path;
            CefString(&settings.cache_path).FromWString(cache_path);
        }

        return CefInitialize(
            main_args, settings, CefRefPtr<CefApp>(new renderer_application(gpu.first, gpu.second)), nullptr);
    });

    if (!result) {
        CASPAR_LOG(error) << "[html] Failed to initialize CEF";
        return;
    }

    g_cef_executor->begin_invoke([&] { CefRunMessageLoop(); });
    dependencies.cg_registry->register_cg_producer(
        L"html",
        {L".html"},
        [](const spl::shared_ptr<core::frame_producer>& producer) { return spl::make_shared<html_cg_proxy>(producer); },
        [](const core::frame_producer_dependencies& dependencies, const std::wstring& filename) {
            return html::create_cg_producer(dependencies, {filename});
        },
        false);
}

void uninit()
{
    if (!g_cef_executor)
        return;

    invoke([] { CefQuitMessageLoop(); });
    g_cef_executor->begin_invoke([&] { CefShutdown(); });
    g_cef_executor.reset();
}

class cef_task : public CefTask
{
  private:
    std::promise<void>    promise_;
    std::function<void()> function_;

  public:
    explicit cef_task(std::function<void()> function)
        : function_(std::move(function))
    {
    }

    void Execute() override
    {
        CASPAR_LOG(trace) << "[cef_task] executing task";

        try {
            function_();
            promise_.set_value();
            CASPAR_LOG(trace) << "[cef_task] task succeeded";
        } catch (...) {
            promise_.set_exception(std::current_exception());
            CASPAR_LOG(warning) << "[cef_task] task failed";
        }
    }

    std::future<void> future() { return promise_.get_future(); }

    IMPLEMENT_REFCOUNTING(cef_task);
};

void invoke(const std::function<void()>& func) { begin_invoke(func).get(); }

std::future<void> begin_invoke(const std::function<void()>& func)
{
    CefRefPtr<cef_task> task = new cef_task(func);

    if (CefCurrentlyOn(TID_UI)) {
        // Avoid deadlock.
        task->Execute();
        return task->future();
    }

    if (CefPostTask(TID_UI, task.get())) {
        return task->future();
    }
    CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("[cef_executor] Could not post task"));
}

std::pair<bool, bool> is_gpu_shared_texture_enabled()
{
    const bool enable_gpu            = env::properties().get(L"configuration.html.enable-gpu", false);
    bool       shared_texture_enable = false;

#ifdef WIN32
    if (enable_gpu) {
        auto dev = accelerator::d3d::d3d_device::get_device();
        if (!dev) {
            CASPAR_LOG(warning) << L"Failed to create directX device for cef gpu acceleration";
        } else {
            shared_texture_enable = true;
        }
    }
#else
    // It would be nice to support this on linux, but it needs some investigation and work
    // Test results (March 2026) suggest that linux without shared-texture is more performant than windows with or
    // without
#endif

    return std::make_pair(enable_gpu, shared_texture_enable);
}

} // namespace caspar::html
