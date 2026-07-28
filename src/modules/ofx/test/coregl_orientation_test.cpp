/*
 * Core-profile OpenGL OFX test plug-in for CasparCG's OFX host.
 *
 * Unlike the bundled OpenGL example (which uses fixed-function glBegin/glOrtho and therefore only
 * runs in a compatibility context), this plug-in renders with ONLY core-profile-valid GL calls
 * (glScissor + glClear). It generates no GL errors in a core context, so CasparCG's OFX host runs
 * it through the true zero-copy path instead of the compatibility fallback.
 *
 * It ignores the source and clears a deterministic, orientation-revealing pattern into the
 * host-bound output framebuffer:
 *     bottom half (framebuffer y = 0 .. h/2)  -> RED
 *     top half    (framebuffer y = h/2 .. h)  -> GREEN
 * After the host's vertical flip to the mixer's top-down convention, a correctly-oriented result
 * shows GREEN at the top of the frame and RED at the bottom (matching the self-contained readback
 * path). A missing/incorrect flip shows the colours swapped.
 *
 * Raw C OFX API (no Support library), so it defines its own entry points and links only opengl32.
 */

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxProperty.h"
#include "ofxParam.h"
#include "ofxGPURender.h"
#include "ofxMultiThread.h"

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <GL/gl.h>
#define EXPORT OfxExport
#else
#include <GL/gl.h>
#define EXPORT __attribute__((visibility("default")))
#endif

static OfxHost*               gHost        = nullptr;
static OfxImageEffectSuiteV1* gEffectHost  = nullptr;
static OfxPropertySuiteV1*    gPropHost    = nullptr;
static OfxParameterSuiteV1*   gParamHost   = nullptr;
static OfxMultiThreadSuiteV1* gThreadHost  = nullptr;

static OfxStatus fetchSuites()
{
    if (!gHost)
        return kOfxStatErrMissingHostFeature;
    gEffectHost = (OfxImageEffectSuiteV1*)gHost->fetchSuite(gHost->host, kOfxImageEffectSuite, 1);
    gPropHost   = (OfxPropertySuiteV1*)gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1);
    gParamHost  = (OfxParameterSuiteV1*)gHost->fetchSuite(gHost->host, kOfxParameterSuite, 1);
    gThreadHost = (OfxMultiThreadSuiteV1*)gHost->fetchSuite(gHost->host, kOfxMultiThreadSuite, 1);
    if (!gEffectHost || !gPropHost || !gParamHost)
        return kOfxStatErrMissingHostFeature;
    return kOfxStatOK;
}

// Multi-thread suite probe: each worker records that its index ran (separate slots, no races).
static unsigned int g_worker_hits[64];
static void mt_worker(unsigned int threadIndex, unsigned int /*threadMax*/, void* /*arg*/)
{
    if (threadIndex < 64)
        g_worker_hits[threadIndex] = 1;
}

static OfxStatus describe(OfxImageEffectHandle effect)
{
    OfxStatus stat = fetchSuites();
    if (stat != kOfxStatOK)
        return stat;

    OfxPropertySetHandle props;
    gEffectHost->getPropertySet(effect, &props);

    gPropHost->propSetString(props, kOfxPropLabel, 0, "Caspar Core-GL Orientation Test");
    gPropHost->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "OFX Test");

    // Declare ONLY the General context (not Filter) so this doubles as a test of the host's
    // General-context support: a General-only plug-in that uses the standard Source/Output clips
    // must be instantiable and renderable just like a filter.
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextGeneral);

    // 8-bit only, so the host negotiates an 8-bit working depth (the zero-copy fast path).
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthByte);

    // OpenGL render supported, CPU not.
    gPropHost->propSetString(props, kOfxImageEffectPropOpenGLRenderSupported, 0, "true");
    gPropHost->propSetString(props, kOfxImageEffectPropCPURenderSupported, 0, "false");
    gPropHost->propSetString(props, kOfxOpenGLPropPixelDepth, 0, kOfxBitDepthByte);

    return kOfxStatOK;
}

static OfxStatus describeInContext(OfxImageEffectHandle effect, OfxPropertySetHandle /*inArgs*/)
{
    OfxPropertySetHandle props;

    gEffectHost->clipDefine(effect, kOfxImageEffectOutputClipName, &props);
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);

    gEffectHost->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &props);
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);

    return kOfxStatOK;
}

static OfxStatus render(OfxImageEffectHandle /*instance*/, OfxPropertySetHandle inArgs, OfxPropertySetHandle /*outArgs*/)
{
    int gl_enabled = 0;
    gPropHost->propGetInt(inArgs, kOfxImageEffectPropOpenGLEnabled, 0, &gl_enabled);
    if (!gl_enabled)
        return kOfxStatErrImageFormat;

    OfxRectI rw;
    gPropHost->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, &rw.x1);
    const int w = rw.x2 - rw.x1;
    const int h = rw.y2 - rw.y1;
    if (w <= 0 || h <= 0)
        return kOfxStatFailed;

    // Probe the host multi-thread suite (no GL off-thread — workers only record their index).
    if (gThreadHost) {
        unsigned int ncpus = 1;
        gThreadHost->multiThreadNumCPUs(&ncpus);
        std::memset(g_worker_hits, 0, sizeof(g_worker_hits));
        gThreadHost->multiThread(mt_worker, ncpus, nullptr);
        unsigned int ran = 0;
        for (unsigned int i = 0; i < 64; ++i)
            ran += g_worker_hits[i];
        static bool logged = false;
        if (!logged) {
            logged = true;
            std::fprintf(stderr, "coregl: multiThread numCPUs=%u workersRan=%u\n", ncpus, ran);
            std::fflush(stderr);
        }
    }

    // Deterministic, orientation-revealing pattern using only core-profile-valid GL. The host has
    // already bound the output texture as the current framebuffer and set the viewport.
    glEnable(GL_SCISSOR_TEST);

    glScissor(0, 0, w, h / 2);            // framebuffer bottom half
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f); // red
    glClear(GL_COLOR_BUFFER_BIT);

    glScissor(0, h / 2, w, h - h / 2);    // framebuffer top half
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f); // green
    glClear(GL_COLOR_BUFFER_BIT);

    glDisable(GL_SCISSOR_TEST);

    return kOfxStatOK;
}

static OfxStatus pluginMain(const char* action, const void* handle, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs)
{
    OfxImageEffectHandle effect = (OfxImageEffectHandle)handle;

    if (strcmp(action, kOfxActionDescribe) == 0)
        return describe(effect);
    if (strcmp(action, kOfxImageEffectActionDescribeInContext) == 0)
        return describeInContext(effect, inArgs);
    if (strcmp(action, kOfxImageEffectActionRender) == 0)
        return render(effect, inArgs, outArgs);

    return kOfxStatReplyDefault;
}

static void setHostFunc(OfxHost* hostStruct) { gHost = hostStruct; }

static OfxPlugin gPlugin = {
    kOfxImageEffectPluginApi,
    1,
    "caspar.test:CoreGLOrientation",
    1,
    0,
    setHostFunc,
    pluginMain};

EXPORT OfxPlugin* OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }
EXPORT int        OfxGetNumberOfPlugins(void) { return 1; }
