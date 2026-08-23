# prores_ks_vulkan: a fixed quantiser (`-q:v`) produces a corrupt picture

`prores_ks_vulkan` with any non-zero `global_quality` writes a file whose chroma planes are lost
and whose luma is clamped to roughly half range. The decoder rejects every frame with
`invalid plane data size`. The same option on the CPU `prores_ks` encoder behaves correctly, so
this looks specific to the Vulkan encoder's `force_quant` path.

Reproduced on FFmpeg 8.1.1 and the 8.1.2 shared Windows build, NVIDIA RTX A4000, driver 582.53,
Vulkan 1.4.312.

## Reproducer

```sh
# correct
ffmpeg -y -init_hw_device vulkan=vk:0 -filter_hw_device vk \
       -f lavfi -i "smptehdbars=size=1920x1080:rate=25:duration=1" \
       -vf "format=rgba64,hwupload,libplacebo=format=yuv422p10" \
       -c:v prores_ks_vulkan -profile:v 3 default.mov

# corrupt -- the only difference is -q:v
ffmpeg -y -init_hw_device vulkan=vk:0 -filter_hw_device vk \
       -f lavfi -i "smptehdbars=size=1920x1080:rate=25:duration=1" \
       -vf "format=rgba64,hwupload,libplacebo=format=yuv422p10" \
       -c:v prores_ks_vulkan -profile:v 3 -q:v 4 forced.mov

ffmpeg -i forced.mov -f null -      # invalid plane data size, every frame
```

## Result

Mean RGB of the first decoded frame, one second of `smptehdbars`:

| encoder | `-q:v` | size | mean R, G, B | max sample |
| :--- | :--- | ---: | :--- | ---: |
| `prores_ks` (CPU) | — | 2.27 MB | 101.5, 105.7, 102.9 | 255 |
| `prores_ks` (CPU) | 4 | 1.80 MB | 101.5, 105.7, 102.9 | 255 |
| `prores_ks_vulkan` | — | 4.50 MB | 100.8, 105.3, 102.3 | 255 |
| `prores_ks_vulkan` | 4 | 1.84 MB | **0.1, 135.0, 0.1** | **135** |

The CPU encoder does what a fixed quantiser should: a smaller file, the same picture. The Vulkan
encoder returns a green frame — red and blue at zero, luma clamped — and a bitstream the decoder
will not accept. The same `0.1, 135.0, 0.1` appears on a real 1080p source as well, so it is not
an artefact of the synthetic pattern.

Two further observations that may help narrow it:

* **The quantiser value barely changes the decoded result while the file size tracks it.** `-q:v`
  2, 4 and 8 produce distinctly different file sizes and decoded frames within 0.1 LSB of one
  another. The value is reaching the encoder; the slices it writes are wrong.
* **`-bits_per_mb` is unaffected.** `-bits_per_mb 950` decodes correctly and matches the default
  in mean and range. It also does not set `force_quant`, which is consistent with the fault being
  in that path specifically.

## Where it might be

`ctx->force_quant` comes from `avctx->global_quality / FF_QP2LAMBDA` in
`proresenc_kostya_common.c`, and `proresenc_kostya_vulkan.c` feeds it to the shaders as
specialisation constants:

```c
SPEC_LIST_ADD(sl, 5, 32, pv->ctx.force_quant ? 0 : pv->ctx.profile_info->min_quant);
SPEC_LIST_ADD(sl, 6, 32, pv->ctx.force_quant ? 0 : pv->ctx.profile_info->max_quant);
...
SPEC_LIST_ADD(sl, 3, 32, pv->ctx.force_quant);
SPEC_LIST_ADD(sl, 5, 32, pv->ctx.force_quant ? pv->ctx.force_quant
                                             : pv->ctx.profile_info->max_quant);
```

and the slice count branches on it:

```c
ctx->force_quant ? 1 : (max_quant - min_quant + 1)
```

`min_quant` and `max_quant` are forced to **0** when `force_quant` is set, while the trellis
shader takes `force_quant` at constant id 3 and short-circuits on `if (force_quant == 0)`
(`prores_ks_trellis_node.comp.glsl`). A quantiser range of `[0, 0]` together with that bypass
seems a plausible route to slices whose written length disagrees with the header, which is what
`invalid plane data size` reports — but I have not confirmed which of these is actually at fault.

## A second, separate defect: `-flags +ildct` hangs

Unrelated to the quantiser, found while measuring data rates per video mode, and reported here
because it is the same encoder.

```sh
ffmpeg -y -init_hw_device vulkan=vk:0 -filter_hw_device vk \
       -f lavfi -i "smptehdbars=size=1920x1080:rate=25:duration=1" \
       -vf "format=rgba64,hwupload,libplacebo=format=yuv422p10" \
       -c:v prores_ks_vulkan -profile:v 3 -flags +ildct out.mov
```

never terminates. Killed after 30 s it leaves a 36-byte file — an empty container — and prints
nothing at any log level: no warning, no "not supported", no error. The software encoder takes the
same flag on the same input and produces a correct file:

| encoder | `+ildct` | result |
| :--- | :--- | :--- |
| `prores_ks` | yes | 9.53 MB for 10 frames, 934 bits/MB (0.98x the profile target) |
| `prores_ks_vulkan` | yes | **hangs**; 36-byte output, no diagnostic |
| `prores_ks_vulkan` | no | correct |

`ff_prores_kostya_encode_init()` sets `pictures_per_frame = 1 + interlaced` and halves `mb_height`
for the interlaced case, and it selects `ff_prores_interlaced_scan`. The Vulkan encoder's dispatch
sizes and its `slices_per_picture` specialisation constants are derived from those, so a shape
mismatch there is the obvious place to look — I have not traced which stage stops making progress.

**Refusing it cleanly would be a strict improvement** even without support: an encoder that
declines with a message costs the caller one error, where one that hangs costs a timeout and a
diagnosis. If interlaced is simply out of scope for this encoder, `AVERROR(ENOSYS)` at init when
`AV_CODEC_FLAG_INTERLACED_DCT` is set would be worth having on its own.

## Why it is easy to miss

The option is a large speed win where the trellis search has real work to do, and the corruption
is silent: the encode succeeds, no warning is printed, and nothing fails until something decodes
the output.

Measured here, 500 frames of 1080p through the same filter chain:

| source | default | `-q:v 4` |
| :--- | :--- | :--- |
| `smptehdbars` | 3.2 s, 90.1 MB | 3.4 s, 36.8 MB |
| moving noise | 10.7 s (46.9 fps), 474 MB | **4.5 s (111.2 fps)**, 1043 MB |

Note that the effect is content-dependent in both directions: on flat bars there is no speed gain
at all and the corrupt file is *smaller*, while on detailed content the encode is 2.4x faster and
the corrupt file is more than twice the size of the correct one. Neither the timing nor the file
size is a reliable signal that something is wrong.
