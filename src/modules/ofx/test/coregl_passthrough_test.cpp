/*
 * Core-profile OpenGL OFX **source-sampling** test plug-in for CasparCG's OFX host.
 *
 * Unlike the orientation test (which clears the framebuffer and ignores the source), this plug-in
 * SAMPLES the host-provided source texture with a core-profile shader and writes it to the bound
 * output framebuffer (identity passthrough). It therefore validates the full zero-copy *source*
 * path — the host's swizzle/flip/premultiply into the source texture — end to end: feed a known
 * pattern and the output must reproduce it.
 *
 * Core-profile only (shaders + VAO, no fixed function), so CasparCG's OFX host runs it through the
 * true zero-copy path. Uses GLEW for the GL 3.3 entry points; links opengl32 + GLEW.
 */

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxProperty.h"
#include "ofxParam.h"
#include "ofxGPURender.h"

#include <GL/glew.h>

#include <cstdio>
#include <cstring>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#define EXPORT OfxExport
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static OfxHost*               gHost       = nullptr;
static OfxImageEffectSuiteV1* gEffectHost = nullptr;
static OfxPropertySuiteV1*    gPropHost   = nullptr;
static OfxImageEffectOpenGLRenderSuiteV1* gGLHost = nullptr;

static GLuint gProgram = 0;
static GLuint gVao     = 0;

static OfxStatus fetchSuites()
{
    if (!gHost)
        return kOfxStatErrMissingHostFeature;
    gEffectHost = (OfxImageEffectSuiteV1*)gHost->fetchSuite(gHost->host, kOfxImageEffectSuite, 1);
    gPropHost   = (OfxPropertySuiteV1*)gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1);
    gGLHost     = (OfxImageEffectOpenGLRenderSuiteV1*)gHost->fetchSuite(gHost->host, kOfxOpenGLRenderSuite, 1);
    if (!gEffectHost || !gPropHost || !gGLHost)
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
    gPropHost->propSetString(props, kOfxPropLabel, 0, "Caspar Core-GL Passthrough Test");
    gPropHost->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "OFX Test");
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
    gPropHost->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthByte);
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

static GLuint compile(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "coregl_pass: shader compile failed: %s\n", log);
        std::fflush(stderr);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static bool ensure_program()
{
    static std::once_flag glew_once;
    static bool           glew_ok = false;
    std::call_once(glew_once, [] {
        glewExperimental = GL_TRUE;
        glew_ok          = (glewInit() == GLEW_OK);
    });
    if (!glew_ok)
        return false;
    if (gProgram)
        return true;

    static const char* kVs =
        "#version 330 core\n"
        "out vec2 vUV;\n"
        "void main(){\n"
        "  vec2 p = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2));\n"
        "  vUV = p;\n"
        "  gl_Position = vec4(p*2.0-1.0, 0.0, 1.0);\n"
        "}\n";
    static const char* kFs =
        "#version 330 core\n"
        "in vec2 vUV;\n"
        "out vec4 o;\n"
        "uniform sampler2D uSrc;\n"
        "void main(){ o = texture(uSrc, vUV); }\n";

    GLuint vs = compile(GL_VERTEX_SHADER, kVs);
    GLuint fs = compile(GL_FRAGMENT_SHADER, kFs);
    if (!vs || !fs)
        return false;
    gProgram = glCreateProgram();
    glAttachShader(gProgram, vs);
    glAttachShader(gProgram, fs);
    glLinkProgram(gProgram);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = GL_FALSE;
    glGetProgramiv(gProgram, GL_LINK_STATUS, &ok);
    if (!ok) {
        glDeleteProgram(gProgram);
        gProgram = 0;
        return false;
    }
    glGenVertexArrays(1, &gVao);
    return true;
}

static OfxStatus render(OfxImageEffectHandle instance, OfxPropertySetHandle inArgs, OfxPropertySetHandle /*outArgs*/)
{
    int gl_enabled = 0;
    gPropHost->propGetInt(inArgs, kOfxImageEffectPropOpenGLEnabled, 0, &gl_enabled);
    if (!gl_enabled)
        return kOfxStatErrImageFormat;

    OfxTime time = 0.0;
    gPropHost->propGetDouble(inArgs, kOfxPropTime, 0, &time);

    // Fetch the host-provided source texture.
    OfxImageClipHandle srcClip = nullptr;
    gEffectHost->clipGetHandle(instance, kOfxImageEffectSimpleSourceClipName, &srcClip, nullptr);
    if (!srcClip)
        return kOfxStatFailed;
    OfxPropertySetHandle srcTex = nullptr;
    if (gGLHost->clipLoadTexture(srcClip, time, nullptr, nullptr, &srcTex) != kOfxStatOK || !srcTex)
        return kOfxStatFailed;
    int srcId = 0;
    gPropHost->propGetInt(srcTex, kOfxImageEffectPropOpenGLTextureIndex, 0, &srcId);

    if (!ensure_program()) {
        gGLHost->clipFreeTexture(srcTex);
        return kOfxStatFailed;
    }

    // The host has bound the output texture as the framebuffer and set the viewport.
    glDisable(GL_BLEND);
    glUseProgram(gProgram);
    glBindVertexArray(gVao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(srcId));
    glUniform1i(glGetUniformLocation(gProgram, "uSrc"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glUseProgram(0);

    gGLHost->clipFreeTexture(srcTex);
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
    kOfxImageEffectPluginApi, 1, "caspar.test:CoreGLPassthrough", 1, 0, setHostFunc, pluginMain};

EXPORT OfxPlugin* OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }
EXPORT int        OfxGetNumberOfPlugins(void) { return 1; }
