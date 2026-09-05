#pragma once

#include <core/mixer/image/blend_modes.h>

#include <common/memory.h>

#include <core/frame/frame_transform.h>
#include <core/frame/geometry.h>
#include <core/frame/pixel_format.h>

#include <utility>

#include "matrix.h"

namespace caspar::accelerator::vulkan {

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

} // namespace caspar::accelerator::vulkan
