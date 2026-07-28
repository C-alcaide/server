/*
 * CUDA test plug-in for CasparCG's OFX host.
 *
 * Validates the host's CUDA render path (kOfxImageEffectPropCudaEnabled): in CUDA mode the images
 * the plug-in fetches carry CUDA *device* pointers in kOfxImagePropData. This plug-in writes a
 * distinctive constant (gray 64) into the output device buffer with cudaMemset — a value that is
 * NOT what a CPU/no-CUDA fallback would produce (the source passes through unchanged), so a gray-64
 * result proves the CUDA path actually ran on a valid output device pointer.
 *
 * Uses only the CUDA runtime host API (cudaMemset), so it needs no .cu/nvcc — just links cudart.
 */

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxProperty.h"
#include "ofxParam.h"
#include "ofxGPURender.h"

#include <cuda_runtime.h>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#define EXPORT OfxExport
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static OfxHost*               gHost       = nullptr;
static OfxImageEffectSuiteV1* gEffectHost = nullptr;
static OfxPropertySuiteV1*    gPropHost   = nullptr;
static OfxParameterSuiteV1*   gParamHost  = nullptr;

struct MyData
{
    OfxImageClipHandle sourceClip = nullptr;
    OfxImageClipHandle outputClip = nullptr;
};

static MyData* getMyData(OfxImageEffectHandle effect)
{
    OfxPropertySetHandle props;
    gEffectHost->getPropertySet(effect, &props);
    MyData* d = nullptr;
    gPropHost->propGetPointer(props, kOfxPropInstanceData, 0, (void**)&d);
    return d;
}

static OfxStatus fetchSuites()
{
    if (!gHost)
        return kOfxStatErrMissingHostFeature;
    gEffectHost = (OfxImageEffectSuiteV1*)gHost->fetchSuite(gHost->host, kOfxImageEffectSuite, 1);
    gPropHost   = (OfxPropertySuiteV1*)gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1);
    gParamHost  = (OfxParameterSuiteV1*)gHost->fetchSuite(gHost->host, kOfxParameterSuite, 1);
    if (!gEffectHost || !gPropHost || !gParamHost)
        return kOfxStatErrMissingHostFeature;
    return kOfxStatOK;
}

static OfxStatus describe(OfxImageEffectHandle effect)
{
    OfxStatus stat = fetchSuites();
    if (stat != kOfxStatOK)
        return stat;

    OfxPropertySetHandle props;
    gEffectHost->getPropertySet(effect, &props);
    gPropHost->propSetString(props, kOfxPropLabel, 0, "Caspar CUDA Test");
    gPropHost->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "OFX Test");
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthByte);

    // CUDA render supported, CPU render not — forces the host's CUDA path.
    gPropHost->propSetString(props, kOfxImageEffectPropCudaRenderSupported, 0, "true");
    gPropHost->propSetString(props, kOfxImageEffectPropCPURenderSupported, 0, "false");
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

static OfxStatus createInstance(OfxImageEffectHandle effect)
{
    MyData* d = new MyData;
    gEffectHost->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &d->sourceClip, nullptr);
    gEffectHost->clipGetHandle(effect, kOfxImageEffectOutputClipName, &d->outputClip, nullptr);
    OfxPropertySetHandle props;
    gEffectHost->getPropertySet(effect, &props);
    gPropHost->propSetPointer(props, kOfxPropInstanceData, 0, (void*)d);
    return kOfxStatOK;
}

static OfxStatus destroyInstance(OfxImageEffectHandle effect)
{
    delete getMyData(effect);
    return kOfxStatOK;
}

static OfxStatus render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle /*outArgs*/)
{
    int cudaEnabled = 0;
    gPropHost->propGetInt(inArgs, kOfxImageEffectPropCudaEnabled, 0, &cudaEnabled);
    if (!cudaEnabled)
        return kOfxStatErrImageFormat; // CUDA-only

    OfxTime time;
    gPropHost->propGetDouble(inArgs, kOfxPropTime, 0, &time);

    MyData* d = getMyData(effect);
    if (!d)
        return kOfxStatFailed;

    OfxPropertySetHandle outImg = nullptr;
    gEffectHost->clipGetImage(d->outputClip, time, nullptr, &outImg);
    if (!outImg)
        return kOfxStatFailed;

    void* dptr    = nullptr;
    int   rowbytes = 0;
    OfxRectI bounds{0, 0, 0, 0};
    gPropHost->propGetPointer(outImg, kOfxImagePropData, 0, &dptr);
    gPropHost->propGetInt(outImg, kOfxImagePropRowBytes, 0, &rowbytes);
    gPropHost->propGetIntN(outImg, kOfxImagePropBounds, 4, &bounds.x1);

    OfxStatus result = kOfxStatOK;
    if (dptr && rowbytes > 0) {
        const int    h    = bounds.y2 - bounds.y1;
        const size_t size = (size_t)rowbytes * (h > 0 ? h : 1);
        // Write a distinctive constant (0x40 = 64) into every byte of the output device buffer.
        cudaError_t e = cudaMemset(dptr, 0x40, size);
        if (e == cudaSuccess)
            e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            result = kOfxStatFailed;
    } else {
        result = kOfxStatFailed;
    }

    gEffectHost->clipReleaseImage(outImg);
    return result;
}

static OfxStatus pluginMain(const char* action, const void* handle, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs)
{
    OfxImageEffectHandle effect = (OfxImageEffectHandle)handle;
    if (strcmp(action, kOfxActionDescribe) == 0)
        return describe(effect);
    if (strcmp(action, kOfxImageEffectActionDescribeInContext) == 0)
        return describeInContext(effect, inArgs);
    if (strcmp(action, kOfxActionCreateInstance) == 0)
        return createInstance(effect);
    if (strcmp(action, kOfxActionDestroyInstance) == 0)
        return destroyInstance(effect);
    if (strcmp(action, kOfxImageEffectActionRender) == 0)
        return render(effect, inArgs, outArgs);
    return kOfxStatReplyDefault;
}

static void setHostFunc(OfxHost* hostStruct) { gHost = hostStruct; }

static OfxPlugin gPlugin = {
    kOfxImageEffectPluginApi,
    1,
    "caspar.test:CudaFill",
    1,
    0,
    setHostFunc,
    pluginMain};

EXPORT OfxPlugin* OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }
EXPORT int        OfxGetNumberOfPlugins(void) { return 1; }
