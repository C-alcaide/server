#pragma once

#include <core/mixer/image/blend_modes.h>

#include <common/memory.h>

#include <core/frame/frame_transform.h>
#include <core/frame/geometry.h>
#include <core/frame/pixel_format.h>

#include <array>
#include <utility>

#include "matrix.h"

namespace caspar::accelerator::ogl {

/// This backend's hand-written layer composition.
///
/// Declared here only so `run_compose_self_test` can pass it to the registry's generated
/// equivalent. Nothing else should call it -- `combine_transform` is the entry point.
void apply_transform_colour_values(core::image_transform& self, const core::image_transform& other);

/// Compare this backend's composition against `core::fields::compose_colour` on randomised
/// transform pairs, and log the result.
///
/// The point is NOT that the generated version is used -- it is not, yet. The point is that
/// the registry's declared rules and the hand-written table are checked against each other
/// on every start, so the two cannot quietly diverge in the window before the swap. A
/// divergence names the FIELD, so the report says which rule disagrees rather than only
/// that one does.
void run_compose_self_test();

/// Check that `source_uv_inverse` returns the matrix the SHADERS use, by round-tripping the
/// default quad through the real `transform_coords`. Fatal on failure -- see transforms.cpp for
/// why this one throws where `run_compose_self_test` warns.
void run_node_uv_self_test();

struct draw_crop_region
{
    explicit draw_crop_region(double left, double top, double right, double bottom);

    void apply_transform(const t_matrix& matrix);

    std::array<t_point, 4> coords;
};

struct draw_transform_step
{
    draw_transform_step()
        : vertex_matrix(boost::numeric::ublas::identity_matrix<double>(3, 3))
    {
    }

    draw_transform_step(const core::corners& perspective, const t_matrix& vertex_matrix)
        : perspective(perspective)
        , vertex_matrix(vertex_matrix)
    {
    }

    core::corners perspective;

    std::vector<draw_crop_region> crop_regions;

    t_matrix vertex_matrix;
};

struct draw_transforms
{
    std::vector<draw_transform_step> steps;

    draw_transforms()
        : image_transform(core::image_transform())
        , steps({draw_transform_step()})
    {
    }

    explicit draw_transforms(core::image_transform transform, std::vector<draw_transform_step> steps)
        : image_transform(transform)
        , steps(std::move(steps))
    {
    }

    core::image_transform image_transform;

    draw_transform_step& current() { return steps.back(); }

    [[nodiscard]] draw_transforms combine_transform(const core::image_transform& transform, double aspect_ratio) const;

    [[nodiscard]] std::vector<core::frame_geometry::coord>
    transform_coords(const std::vector<core::frame_geometry::coord>& coords) const;
};


/// See transforms.cpp for the account of both of these.
draw_transforms apply_geometry_scale_mode(const draw_transforms&      transforms,
                                          const core::frame_geometry& geometry,
                                          int                         target_width,
                                          int                         target_height,
                                          int                         plane_width,
                                          int                         plane_height,
                                          double                      aspect_ratio);

bool source_uv_inverse(const draw_transforms& transforms, std::array<float, 9>& out);

} // namespace caspar::accelerator::ogl