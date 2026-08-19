import subprocess, numpy as np, itertools, sys

W = H = 16
src = np.fromfile('in_bgra64le.raw', dtype='<u2').reshape(-1, 4)   # B,G,R,A
want = src[:, [2, 1, 0, 3]].astype(np.int64)                        # R,G,B,A

def run(inraw, inpix, outpix, extra):
    cmd = ['ffmpeg', '-v', 'error', '-y',
           '-f', 'rawvideo', '-pix_fmt', inpix, '-s', f'{W}x{H}', '-i', inraw] + extra + \
          ['-f', 'rawvideo', '-pix_fmt', outpix, '-frames:v', '1', '-']
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        return None, r.stderr.decode(errors='replace').strip()[:160]
    dt = '>u2' if outpix.endswith('be') else '<u2'
    a = np.frombuffer(r.stdout, dtype=dt).reshape(-1, 4).astype(np.int64)
    return a, None

CASES = [
  ("direct  default",            ['-pix_fmt']),
  ("direct  accurate_rnd",       ['-sws_flags', 'accurate_rnd', '-pix_fmt']),
  ("direct  acc+full_chr",       ['-sws_flags', 'accurate_rnd+full_chroma_int+full_chroma_inp', '-pix_fmt']),
  ("direct  bitexact",           ['-sws_flags', 'bitexact', '-pix_fmt']),
  ("direct  point",              ['-sws_flags', 'point', '-pix_fmt']),
  ("direct  dither=none",        ['-sws_dither', 'none', '-pix_fmt']),
  ("direct  dither=none+acc",    ['-sws_dither', 'none', '-sws_flags', 'accurate_rnd', '-pix_fmt']),
  ("direct  dither=none+bitex",  ['-sws_dither', 'none', '-sws_flags', 'bitexact+accurate_rnd', '-pix_fmt']),
  ("direct  dither=bayer",       ['-sws_dither', 'bayer', '-pix_fmt']),
  ("direct  dither=a_dither",    ['-sws_dither', 'a_dither', '-pix_fmt']),
]
print("=== bgra64le -> rgba64be, direct ===")
for name, pre in CASES:
    got, err = run('in_bgra64le.raw', 'bgra64le', 'rgba64be', pre[:-1])
    if got is None:
        print(f"  {name:26s} ERROR {err}"); continue
    d = got - want
    print(f"  {name:26s} maxabs={np.abs(d).max():6d}  nonzero={int((d!=0).sum()):4d}/1024  "
          f"{'EXACT' if not d.any() else 'lossy'}")

print()
print("=== controls and alternate routes ===")
ROUTES = [
  ("rgba64le -> rgba64be (Vulkan arm)", 'in_rgba64le.raw', 'rgba64le', 'rgba64be', [], src[:, [2,1,0,3]]),
  ("bgra64le -> bgra64le (identity)",   'in_bgra64le.raw', 'bgra64le', 'bgra64le', [], src),
  ("bgra64le -> bgra64be (endian only)",'in_bgra64le.raw', 'bgra64le', 'bgra64be', [], src),
  ("bgra64le -> rgba64le (perm only)",  'in_bgra64le.raw', 'bgra64le', 'rgba64le', [], src[:, [2,1,0,3]]),
  ("bgra64le -> gbrap16le -> rgba64be", 'in_bgra64le.raw', 'bgra64le', 'rgba64be',
       ['-vf', 'format=gbrap16le'], src[:, [2,1,0,3]]),
  ("bgra64le -> rgba64le -> rgba64be",  'in_bgra64le.raw', 'bgra64le', 'rgba64be',
       ['-vf', 'format=rgba64le'], src[:, [2,1,0,3]]),
  ("bgra64le -> gbrap16le (planar)",    'in_bgra64le.raw', 'bgra64le', 'gbrap16le', [], None),
]
for name, inr, ip, op, extra, exp in ROUTES:
    got, err = run(inr, ip, op, extra)
    if got is None:
        print(f"  {name:38s} ERROR {err}"); continue
    if exp is None:
        print(f"  {name:38s} (planar, shape {got.shape}) skipped"); continue
    d = got - exp.astype(np.int64)
    print(f"  {name:38s} maxabs={np.abs(d).max():6d}  {'EXACT' if not d.any() else 'lossy'}")
