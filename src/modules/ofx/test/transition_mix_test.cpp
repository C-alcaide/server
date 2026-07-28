/*
 * CPU transition test plug-in for CasparCG's OFX host (Transition context).
 *
 * Blends the two mandatory transition inputs by the mandatory "Transition" parameter:
 *     Output = SourceFrom * (1 - t) + SourceTo * t
 * Used to validate the host's transition-context support (dual SourceFrom/SourceTo clips + the
 * host-driven Transition parameter). Raw C OFX API, CPU render, 8-bit RGBA.
 */

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxProperty.h"
#include "ofxParam.h"

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
    OfxImageClipHandle fromClip = nullptr;
    OfxImageClipHandle toClip   = nullptr;
    OfxImageClipHandle outClip  = nullptr;
    OfxParamHandle     transition = nullptr;
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
    gPropHost->propSetString(props, kOfxPropLabel, 0, "Caspar Transition Test");
    gPropHost->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "OFX Test");
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextTransition);
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthByte);
    return kOfxStatOK;
}

static OfxStatus describeInContext(OfxImageEffectHandle effect, OfxPropertySetHandle /*inArgs*/)
{
    OfxPropertySetHandle props;

    gEffectHost->clipDefine(effect, kOfxImageEffectTransitionSourceFromClipName, &props);
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);

    gEffectHost->clipDefine(effect, kOfxImageEffectTransitionSourceToClipName, &props);
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);

    gEffectHost->clipDefine(effect, kOfxImageEffectOutputClipName, &props);
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);

    // The mandatory transition parameter (0..1), set by the host each frame.
    OfxParamSetHandle paramSet;
    gEffectHost->getParamSet(effect, &paramSet);
    OfxPropertySetHandle pprops;
    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, kOfxImageEffectTransitionParamName, &pprops);
    gPropHost->propSetDouble(pprops, kOfxParamPropDefault, 0, 0.0);
    gPropHost->propSetDouble(pprops, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pprops, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetString(pprops, kOfxPropLabel, 0, "Transition");
    return kOfxStatOK;
}

static OfxStatus createInstance(OfxImageEffectHandle effect)
{
    MyData* d = new MyData;
    gEffectHost->clipGetHandle(effect, kOfxImageEffectTransitionSourceFromClipName, &d->fromClip, nullptr);
    gEffectHost->clipGetHandle(effect, kOfxImageEffectTransitionSourceToClipName, &d->toClip, nullptr);
    gEffectHost->clipGetHandle(effect, kOfxImageEffectOutputClipName, &d->outClip, nullptr);

    OfxParamSetHandle paramSet;
    gEffectHost->getParamSet(effect, &paramSet);
    gParamHost->paramGetHandle(paramSet, kOfxImageEffectTransitionParamName, &d->transition, nullptr);

    OfxPropertySetHandle props;
    gEffectHost->getPropertySet(effect, &props);
    gPropHost->propSetPointer(props, kOfxPropInstanceData, 0, (void*)d);
    return kOfxStatOK;
}

static OfxStatus destroyInstance(OfxImageEffectHandle effect)
{
    MyData* d = getMyData(effect);
    delete d;
    return kOfxStatOK;
}

static OfxStatus render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle /*outArgs*/)
{
    OfxTime time;
    gPropHost->propGetDouble(inArgs, kOfxPropTime, 0, &time);
    OfxRectI rw;
    gPropHost->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, &rw.x1);

    MyData* d = getMyData(effect);
    if (!d)
        return kOfxStatFailed;

    double t = 0.0;
    gParamHost->paramGetValueAtTime(d->transition, time, &t);
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;

    OfxPropertySetHandle fromImg = nullptr, toImg = nullptr, outImg = nullptr;
    gEffectHost->clipGetImage(d->fromClip, time, nullptr, &fromImg);
    gEffectHost->clipGetImage(d->toClip, time, nullptr, &toImg);
    gEffectHost->clipGetImage(d->outClip, time, nullptr, &outImg);

    OfxStatus result = kOfxStatOK;
    if (fromImg && toImg && outImg) {
        void *fp = nullptr, *tp = nullptr, *op = nullptr;
        int   frb = 0, trb = 0, orb = 0;
        gPropHost->propGetPointer(fromImg, kOfxImagePropData, 0, &fp);
        gPropHost->propGetPointer(toImg, kOfxImagePropData, 0, &tp);
        gPropHost->propGetPointer(outImg, kOfxImagePropData, 0, &op);
        gPropHost->propGetInt(fromImg, kOfxImagePropRowBytes, 0, &frb);
        gPropHost->propGetInt(toImg, kOfxImagePropRowBytes, 0, &trb);
        gPropHost->propGetInt(outImg, kOfxImagePropRowBytes, 0, &orb);

        if (fp && tp && op) {
            for (int y = rw.y1; y < rw.y2; ++y) {
                const unsigned char* fr = (const unsigned char*)fp + (size_t)y * frb;
                const unsigned char* tr = (const unsigned char*)tp + (size_t)y * trb;
                unsigned char*       orow = (unsigned char*)op + (size_t)y * orb;
                for (int x = rw.x1; x < rw.x2; ++x) {
                    for (int c = 0; c < 4; ++c) {
                        const int fv = fr[x * 4 + c];
                        const int tv = tr[x * 4 + c];
                        orow[x * 4 + c] = (unsigned char)(fv + (tv - fv) * t + 0.5);
                    }
                }
            }
        } else {
            result = kOfxStatFailed;
        }
    } else {
        result = kOfxStatFailed;
    }

    if (fromImg) gEffectHost->clipReleaseImage(fromImg);
    if (toImg)   gEffectHost->clipReleaseImage(toImg);
    if (outImg)  gEffectHost->clipReleaseImage(outImg);
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
    "caspar.test:TransitionMix",
    1,
    0,
    setHostFunc,
    pluginMain};

EXPORT OfxPlugin* OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }
EXPORT int        OfxGetNumberOfPlugins(void) { return 1; }
