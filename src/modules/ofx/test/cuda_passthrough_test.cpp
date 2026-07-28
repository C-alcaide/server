/*
 * CUDA **source-sampling** passthrough test plug-in for CasparCG's OFX host.
 *
 * Where CudaFill writes a uniform constant (and so cannot reveal orientation or channel order), this
 * plug-in copies the source device buffer to the output device buffer (identity) with cudaMemcpy2D.
 * Feed a known spatial pattern and the output must reproduce it — validating the host's CUDA *source*
 * convert (swizzle/flip/premultiply) and the overall CUDA-path orientation/channels end to end.
 *
 * CUDA runtime host API only (cudaMemcpy2D) — no .cu/nvcc; links cudart.
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
    if (!gEffectHost || !gPropHost)
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
    gPropHost->propSetString(props, kOfxPropLabel, 0, "Caspar CUDA Passthrough Test");
    gPropHost->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "OFX Test");
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthByte);
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
        return kOfxStatErrImageFormat;

    OfxTime time;
    gPropHost->propGetDouble(inArgs, kOfxPropTime, 0, &time);

    MyData* d = getMyData(effect);
    if (!d)
        return kOfxStatFailed;

    OfxPropertySetHandle srcImg = nullptr;
    OfxPropertySetHandle outImg = nullptr;
    gEffectHost->clipGetImage(d->sourceClip, time, nullptr, &srcImg);
    gEffectHost->clipGetImage(d->outputClip, time, nullptr, &outImg);

    OfxStatus result = kOfxStatFailed;
    if (srcImg && outImg) {
        void*    sptr = nullptr;
        void*    optr = nullptr;
        int      srb = 0, orb = 0;
        OfxRectI ob{0, 0, 0, 0};
        gPropHost->propGetPointer(srcImg, kOfxImagePropData, 0, &sptr);
        gPropHost->propGetInt(srcImg, kOfxImagePropRowBytes, 0, &srb);
        gPropHost->propGetPointer(outImg, kOfxImagePropData, 0, &optr);
        gPropHost->propGetInt(outImg, kOfxImagePropRowBytes, 0, &orb);
        gPropHost->propGetIntN(outImg, kOfxImagePropBounds, 4, &ob.x1);
        const int w = ob.x2 - ob.x1;
        const int h = ob.y2 - ob.y1;
        if (sptr && optr && w > 0 && h > 0) {
            cudaError_t e = cudaMemcpy2D(optr, orb, sptr, srb, (size_t)w * 4, h, cudaMemcpyDeviceToDevice);
            if (e == cudaSuccess)
                e = cudaDeviceSynchronize();
            result = (e == cudaSuccess) ? kOfxStatOK : kOfxStatFailed;
        }
    }
    if (srcImg)
        gEffectHost->clipReleaseImage(srcImg);
    if (outImg)
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
    kOfxImageEffectPluginApi, 1, "caspar.test:CudaPassthrough", 1, 0, setHostFunc, pluginMain};

EXPORT OfxPlugin* OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }
EXPORT int        OfxGetNumberOfPlugins(void) { return 1; }
