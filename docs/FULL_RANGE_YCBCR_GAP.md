# Full-range YCbCr sources are stretched by 255/219

**Found 2026-08-19, measured, not fixed.** Pre-existing and shared with upstream; nothing in the
FFmpeg 8 sync caused it. Recorded because it mis-renders a whole class of source silently.

## What happens

`ycbcra_to_rgba` in both mixers assumes **studio-swing (limited range)** unconditionally:

```glsl
const float luma_coefficient   = 255.0/219.0;
const float chroma_coefficient = 255.0/224.0;
vec3 YCbCr = vec3(Y, Cb, Cr) * ycbcr_code_scale;
YCbCr -= vec3(16.0, 128.0, 128.0);      // limited-range black level
```

A source that is genuinely **full range** (`AVCOL_RANGE_JPEG`) is therefore expanded a second
time: blacks are crushed below 0 and whites clipped above 255, by a factor of 255/219 = 1.1644.

## Measured

A 12288x6144 NotchLC asset, whose FFmpeg decoder declares `AVCOL_RANGE_JPEG`
(`libavcodec/notchlc.c:67`). Three paths, one frame, fitted against an **independent `ffmpeg`
CLI decode outside CasparCG** as ground truth:

| path | fit vs ground truth | residual | verdict |
| :--- | :--- | ---: | :--- |
| `CUDA_NOTCHLC` producer (RGB out, no shader YCbCr step) | s=0.9993 o=-0.41 | 0.79 | **correct** |
| FFmpeg producer (YUV out, shader converts) | s=1.1683 o=-19.09 | 0.80 | **stretched** |

s=1.1683 against the theoretical 255/219=1.1644 is a 0.3% match, and the residual after fitting
the range drops from 6.06 to 0.32 -- so the range accounts for essentially all of it. Channel
order is correct on both paths (straight fits better than R/B-swapped).

## Why the information cannot reach the shader

`av_producer` **does** observe the source range -- `frame_color_range`, `av_producer.cpp:1650` --
and uses it at line 2265 to configure the filtergraph's `buffersrc`. That is where it stops.
`core::pixel_format_desc` carries `color_space` and `color_transfer` and **no range field**, so
nothing downstream of the producer knows, and the shader has no parameter to switch on.

## Scope

* Any full-range YCbCr source through the FFmpeg producer: NotchLC, MJPEG, JPEG-range camera
  material.
* **Not** RGB sources -- no YCbCr step to get wrong. Stills, HTML, colour producers unaffected.
* **Not** the `CUDA_NOTCHLC` path, which converts on the GPU with full-range coefficients and
  measures correct.
* Broadcast material, which is limited range and by far the common case, is unaffected -- which
  is why this has survived.

## What fixing it involves, and why it was not done here

This is a cross-cutting change, not a local fix:

1. a range field on `core::pixel_format_desc`, defaulting to limited so nothing changes for
   existing content;
2. propagation from `av_producer` (and any other YCbCr producer -- DeckLink, GStreamer) into
   the frame descriptor;
3. a uniform through **both** mixers, and the matching branch in both copies of
   `ycbcra_to_rgba`;
4. the `apply_transform_colour_values` allowlist in **both** `transforms.cpp` if it is carried
   on `image_transform` -- a field absent from that list is silently dropped, which presents as
   a command that returns 202 and changes nothing;
5. a battery: a flat full-range fixture is the honest oracle, since "neutral in, neutral out"
   needs no colour model. `flat-decoded` is the closest existing shape.

The regression risk sits entirely on step 1's default: get it wrong and every existing broadcast
clip shifts. That is not a change to slip in at the end of a session, and it wants its own
before/after on real limited-range material.
