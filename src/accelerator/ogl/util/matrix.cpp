

#include <core/frame/frame_transform.h>

#include <common/except.h>
#include <common/log.h>

#include <cmath>
#include <utility>
#include <vector>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_vector.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include "matrix.h"

namespace caspar::accelerator::ogl {

t_matrix create_matrix(std::vector<std::vector<double>> data)
{
    if (data.empty())
        CASPAR_THROW_EXCEPTION(invalid_argument() << msg_info(L"data cannot be empty"));

    t_matrix matrix(data.size(), data.at(0).size());
    for (int y = 0; y < matrix.size1(); ++y) {
        if (data.at(y).size() != matrix.size2())
            CASPAR_THROW_EXCEPTION(invalid_argument() << msg_info(L"Each row must be of the same length"));

        for (int x = 0; x < matrix.size2(); ++x)
            matrix(x, y) = data.at(y).at(x);
    }
    return matrix;
}

t_matrix get_vertex_matrix(const core::image_transform& transform, double aspect_ratio)
{
    using namespace boost::numeric::ublas;
    auto anchor_matrix =
        create_matrix({{1.0, 0.0, -transform.anchor[0]}, {0.0, 1.0, -transform.anchor[1]}, {0.0, 0.0, 1.0}});
    auto scale_matrix =
        create_matrix({{transform.fill_scale[0], 0.0, 0.0}, {0.0, transform.fill_scale[1], 0.0}, {0.0, 0.0, 1.0}});
    auto aspect_matrix      = create_matrix({{1.0, 0.0, 0.0}, {0.0, 1.0 / aspect_ratio, 0.0}, {0.0, 0.0, 1.0}});
    auto aspect_inv_matrix  = create_matrix({{1.0, 0.0, 0.0}, {0.0, aspect_ratio, 0.0}, {0.0, 0.0, 1.0}});
    auto rotation_matrix    = create_matrix({{std::cos(transform.angle), -std::sin(transform.angle), 0.0},
                                             {std::sin(transform.angle), std::cos(transform.angle), 0.0},
                                             {0.0, 0.0, 1.0}});
    auto translation_matrix = create_matrix(
        {{1.0, 0.0, transform.fill_translation[0]}, {0.0, 1.0, transform.fill_translation[1]}, {0.0, 0.0, 1.0}});

    return anchor_matrix * aspect_matrix * scale_matrix * rotation_matrix * aspect_inv_matrix * translation_matrix;
}


bool invert_3x3(const t_matrix& m, t_matrix& out)
{
    if (m.size1() != 3 || m.size2() != 3)
        return false;

    const double a = m(0, 0), b = m(0, 1), c = m(0, 2);
    const double d = m(1, 0), e = m(1, 1), f = m(1, 2);
    const double g = m(2, 0), h = m(2, 1), i = m(2, 2);

    // The cofactors, which are also the adjugate's columns -- so the transpose is free.
    const double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const double det = a * A + b * B + c * C;

    // NOT `det == 0.0`. `/fp:fast` is on (see CLAUDE.md) and a determinant assembled from nine
    // products is never exactly zero for a degenerate-in-intent transform; a scale of 1e-9 would
    // pass an equality test and then produce a uniform of 1e9.
    if (!std::isfinite(det) || std::abs(det) < 1e-12)
        return false;

    const double D = -(b * i - c * h), E = a * i - c * g, F = -(a * h - b * g);
    const double G = b * f - c * e, H = -(a * f - c * d), I = a * e - b * d;

    out = t_matrix(3, 3);
    out(0, 0) = A / det;
    out(0, 1) = D / det;
    out(0, 2) = G / det;
    out(1, 0) = B / det;
    out(1, 1) = E / det;
    out(1, 2) = H / det;
    out(2, 0) = C / det;
    out(2, 1) = F / det;
    out(2, 2) = I / det;
    return true;
}

} // namespace caspar::accelerator::ogl
