# swscale: packed 16-bit RGB component permutation is lossy and dithers when it must not

**Version:** ffmpeg 8.1.1 (also reproduced on 7.0.2), Windows x86_64, gcc 15.2.0.

## Summary

`bgra64le -> rgba64be` is a pure component permutation plus a byte swap. Neither reduces bit
depth, so it must be lossless. swscale instead returns values deviating by up to **35 LSB16 on
72% of components**, and the deviation is **position-dependent** — the same input sample maps
to different outputs at different pixels, which is dithering. `-sws_dither none` does not
suppress it, nor does `accurate_rnd`, nor `bitexact`.

## Reproduce

```sh
python -c "
import numpy as np
a = np.random.RandomState(11).randint(0,65536,(1920*1080,4)).astype('<u2'); a[:,3]=65535
a[:, [2,1,0,3]].copy().tofile('bgra.raw'); a.tofile('ref_rgba.raw')"

ffmpeg -v error -y -f rawvideo -pix_fmt bgra64le -s 1920x1080 -i bgra.raw \
       -f rawvideo -pix_fmt rgba64be -frames:v 1 out.raw

python -c "
import numpy as np
w = np.fromfile('ref_rgba.raw', dtype='<u2').astype(int)
g = np.fromfile('out.raw',      dtype='>u2').astype(int)
d = g - w
print('maxabs', abs(d).max(), 'nonzero', (d!=0).sum(), 'of', d.size)"
```

Observed: `maxabs 35 nonzero 5937403 of 8294400`. Expected: `maxabs 0 nonzero 0`.

## Scope

| conversion | result |
| :--- | :--- |
| `bgra64le -> bgra64le` | exact |
| `bgra64le -> bgra64be` (endian only) | exact |
| `rgba64le -> rgba64be` (endian only) | exact |
| `bgra64le -> rgba64le` (permutation only) | **lossy, max 32** |
| `bgra64le -> rgba64be` (permutation + endian) | **lossy, max 35** |
| `bgra64le -> gbrap16le -> rgba64be` (via planar) | exact |
| `bgra -> rgba` (the same permutation at 8 bit) | exact |

So the endian-swap and identity fast paths are fine, the 8-bit permutation is fine, and routing
through planar 16-bit is fine. Only the **packed-16 to packed-16 permutation** is affected — it
appears to miss a shuffle fast path and fall into a general path carrying a dither stage that
should be inactive when input and output depths are equal.

Flags tried, all producing byte-identical error: default, `accurate_rnd`,
`accurate_rnd+full_chroma_int+full_chroma_inp`, `bitexact`, `sws_dither=none`,
`sws_dither=none+accurate_rnd`, `sws_dither=none+bitexact+accurate_rnd`, `sws_dither=a_dither`.
`sws_dither=bayer` is worse (max 48281, consistent with wraparound).

## Impact

Found in CasparCG Server, converting a 16-bit BGRA framebuffer readback to `rgba64be` for PNG
encoding. It cost roughly 9 LSB16 of accuracy on every high-bit-depth capture, invisible at 8
bits, which is why it went unnoticed for a long time.
