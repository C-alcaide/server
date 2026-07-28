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
 */

#include "ofx_param_instance.h"

namespace caspar { namespace ofx {

namespace {

double def_d(OFX::Host::Param::Descriptor& d, int i) { return d.getProperties().getDoubleProperty(kOfxParamPropDefault, i); }
int    def_i(OFX::Host::Param::Descriptor& d, int i) { return d.getProperties().getIntProperty(kOfxParamPropDefault, i); }

class integer_param : public OFX::Host::Param::IntegerInstance
{
    int v_;

  public:
    integer_param(OFX::Host::Param::Descriptor& d, OFX::Host::Param::SetInstance* i)
        : OFX::Host::Param::IntegerInstance(d, i)
        , v_(def_i(d, 0))
    {
    }
    OfxStatus get(int& v) override { v = v_; return kOfxStatOK; }
    OfxStatus get(OfxTime, int& v) override { v = v_; return kOfxStatOK; }
    OfxStatus set(int v) override { v_ = v; return kOfxStatOK; }
    OfxStatus set(OfxTime, int v) override { v_ = v; return kOfxStatOK; }
};

class choice_param : public OFX::Host::Param::ChoiceInstance
{
    int v_;

  public:
    choice_param(OFX::Host::Param::Descriptor& d, OFX::Host::Param::SetInstance* i)
        : OFX::Host::Param::ChoiceInstance(d, i)
        , v_(def_i(d, 0))
    {
    }
    OfxStatus get(int& v) override { v = v_; return kOfxStatOK; }
    OfxStatus get(OfxTime, int& v) override { v = v_; return kOfxStatOK; }
    OfxStatus set(int v) override { v_ = v; return kOfxStatOK; }
    OfxStatus set(OfxTime, int v) override { v_ = v; return kOfxStatOK; }
};

class boolean_param : public OFX::Host::Param::BooleanInstance
{
    bool v_;

  public:
    boolean_param(OFX::Host::Param::Descriptor& d, OFX::Host::Param::SetInstance* i)
        : OFX::Host::Param::BooleanInstance(d, i)
        , v_(def_i(d, 0) != 0)
    {
    }
    OfxStatus get(bool& v) override { v = v_; return kOfxStatOK; }
    OfxStatus get(OfxTime, bool& v) override { v = v_; return kOfxStatOK; }
    OfxStatus set(bool v) override { v_ = v; return kOfxStatOK; }
    OfxStatus set(OfxTime, bool v) override { v_ = v; return kOfxStatOK; }
};

class double_param : public OFX::Host::Param::DoubleInstance
{
    double v_;

  public:
    double_param(OFX::Host::Param::Descriptor& d, OFX::Host::Param::SetInstance* i)
        : OFX::Host::Param::DoubleInstance(d, i)
        , v_(def_d(d, 0))
    {
    }
    OfxStatus get(double& v) override { v = v_; return kOfxStatOK; }
    OfxStatus get(OfxTime, double& v) override { v = v_; return kOfxStatOK; }
    OfxStatus set(double v) override { v_ = v; return kOfxStatOK; }
    OfxStatus set(OfxTime, double v) override { v_ = v; return kOfxStatOK; }
    OfxStatus derive(OfxTime, double& v) override { v = 0.0; return kOfxStatOK; }
    OfxStatus integrate(OfxTime, OfxTime, double& v) override { v = 0.0; return kOfxStatOK; }
};

class double2d_param : public OFX::Host::Param::Double2DInstance
{
    double x_, y_;

  public:
    double2d_param(OFX::Host::Param::Descriptor& d, OFX::Host::Param::SetInstance* i)
        : OFX::Host::Param::Double2DInstance(d, i)
        , x_(def_d(d, 0))
        , y_(def_d(d, 1))
    {
    }
    OfxStatus get(double& x, double& y) override { x = x_; y = y_; return kOfxStatOK; }
    OfxStatus get(OfxTime, double& x, double& y) override { x = x_; y = y_; return kOfxStatOK; }
    OfxStatus set(double x, double y) override { x_ = x; y_ = y; return kOfxStatOK; }
    OfxStatus set(OfxTime, double x, double y) override { x_ = x; y_ = y; return kOfxStatOK; }
};

class integer2d_param : public OFX::Host::Param::Integer2DInstance
{
    int x_, y_;

  public:
    integer2d_param(OFX::Host::Param::Descriptor& d, OFX::Host::Param::SetInstance* i)
        : OFX::Host::Param::Integer2DInstance(d, i)
        , x_(def_i(d, 0))
        , y_(def_i(d, 1))
    {
    }
    OfxStatus get(int& x, int& y) override { x = x_; y = y_; return kOfxStatOK; }
    OfxStatus get(OfxTime, int& x, int& y) override { x = x_; y = y_; return kOfxStatOK; }
    OfxStatus set(int x, int y) override { x_ = x; y_ = y; return kOfxStatOK; }
    OfxStatus set(OfxTime, int x, int y) override { x_ = x; y_ = y; return kOfxStatOK; }
};

class rgba_param : public OFX::Host::Param::RGBAInstance
{
    double r_, g_, b_, a_;

  public:
    rgba_param(OFX::Host::Param::Descriptor& d, OFX::Host::Param::SetInstance* i)
        : OFX::Host::Param::RGBAInstance(d, i)
        , r_(def_d(d, 0))
        , g_(def_d(d, 1))
        , b_(def_d(d, 2))
        , a_(def_d(d, 3))
    {
    }
    OfxStatus get(double& r, double& g, double& b, double& a) override { r = r_; g = g_; b = b_; a = a_; return kOfxStatOK; }
    OfxStatus get(OfxTime, double& r, double& g, double& b, double& a) override { r = r_; g = g_; b = b_; a = a_; return kOfxStatOK; }
    OfxStatus set(double r, double g, double b, double a) override { r_ = r; g_ = g; b_ = b; a_ = a; return kOfxStatOK; }
    OfxStatus set(OfxTime, double r, double g, double b, double a) override { r_ = r; g_ = g; b_ = b; a_ = a; return kOfxStatOK; }
};

class rgb_param : public OFX::Host::Param::RGBInstance
{
    double r_, g_, b_;

  public:
    rgb_param(OFX::Host::Param::Descriptor& d, OFX::Host::Param::SetInstance* i)
        : OFX::Host::Param::RGBInstance(d, i)
        , r_(def_d(d, 0))
        , g_(def_d(d, 1))
        , b_(def_d(d, 2))
    {
    }
    OfxStatus get(double& r, double& g, double& b) override { r = r_; g = g_; b = b_; return kOfxStatOK; }
    OfxStatus get(OfxTime, double& r, double& g, double& b) override { r = r_; g = g_; b = b_; return kOfxStatOK; }
    OfxStatus set(double r, double g, double b) override { r_ = r; g_ = g; b_ = b; return kOfxStatOK; }
    OfxStatus set(OfxTime, double r, double g, double b) override { r_ = r; g_ = g; b_ = b; return kOfxStatOK; }
};

class pushbutton_param : public OFX::Host::Param::PushbuttonInstance
{
  public:
    pushbutton_param(OFX::Host::Param::Descriptor& d, OFX::Host::Param::SetInstance* i)
        : OFX::Host::Param::PushbuttonInstance(d, i)
    {
    }
};

class string_param : public OFX::Host::Param::StringInstance
{
    std::string v_;

    static std::string default_of(OFX::Host::Param::Descriptor& d)
    {
        try {
            return d.getProperties().getStringProperty(kOfxParamPropDefault, 0);
        } catch (...) {
            return std::string();
        }
    }

  public:
    string_param(OFX::Host::Param::Descriptor& d, OFX::Host::Param::SetInstance* i)
        : OFX::Host::Param::StringInstance(d, i)
        , v_(default_of(d))
    {
    }
    OfxStatus get(std::string& v) override { v = v_; return kOfxStatOK; }
    OfxStatus get(OfxTime, std::string& v) override { v = v_; return kOfxStatOK; }
    OfxStatus set(const char* v) override { v_ = v ? v : ""; return kOfxStatOK; }
    OfxStatus set(OfxTime, const char* v) override { v_ = v ? v : ""; return kOfxStatOK; }
};

} // namespace

OFX::Host::Param::Instance* create_param_instance(OFX::Host::Param::SetInstance* effect,
                                                  const std::string& /*name*/,
                                                  OFX::Host::Param::Descriptor& d)
{
    const std::string& type = d.getType();

    if (type == kOfxParamTypeInteger)
        return new integer_param(d, effect);
    if (type == kOfxParamTypeDouble)
        return new double_param(d, effect);
    if (type == kOfxParamTypeBoolean)
        return new boolean_param(d, effect);
    if (type == kOfxParamTypeChoice)
        return new choice_param(d, effect);
    if (type == kOfxParamTypeRGBA)
        return new rgba_param(d, effect);
    if (type == kOfxParamTypeRGB)
        return new rgb_param(d, effect);
    if (type == kOfxParamTypeDouble2D)
        return new double2d_param(d, effect);
    if (type == kOfxParamTypeInteger2D)
        return new integer2d_param(d, effect);
    if (type == kOfxParamTypePushButton)
        return new pushbutton_param(d, effect);
    if (type == kOfxParamTypeString)
        return new string_param(d, effect);
    if (type == kOfxParamTypeGroup)
        return new OFX::Host::Param::GroupInstance(d, effect);
    if (type == kOfxParamTypePage)
        return new OFX::Host::Param::PageInstance(d, effect);

    return nullptr;
}

}} // namespace caspar::ofx
