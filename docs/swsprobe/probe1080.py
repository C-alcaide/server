import subprocess, numpy as np
W, H = 1920, 1080
rng = np.random.RandomState(11)
a = rng.randint(0, 65536, (W*H, 4)).astype('<u2')
a[:, 3] = 65535
a[:16] = np.array([[0,0,0,65535],[65535,65535,65535,65535],[0,65535,0,65535],[65535,0,0,65535]]*4, dtype='<u2')
a.tofile('big_rgba64le.raw')       # interpret as R,G,B,A
b = a[:, [2,1,0,3]].copy(); b.tofile('big_bgra64le.raw')

def conv(inraw, ip, op, extra=[]):
    r = subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt',ip,'-s',f'{W}x{H}',
                        '-i',inraw]+extra+['-f','rawvideo','-pix_fmt',op,'-frames:v','1','-'],
                       capture_output=True)
    assert r.returncode == 0, r.stderr.decode()[:300]
    dt = '>u2' if op.endswith('be') else '<u2'
    return np.frombuffer(r.stdout, dtype=dt).reshape(-1,4).astype(np.int64)

want = a.astype(np.int64)
for name, inr, ip, extra in [
    ("rgba64le -> rgba64be   (THE FIX RELIES ON THIS)", 'big_rgba64le.raw', 'rgba64le', []),
    ("bgra64le -> rgba64be   (what it replaces)",       'big_bgra64le.raw', 'bgra64le', []),
    ("bgra64le -> gbrap16le -> rgba64be (alt route)",   'big_bgra64le.raw', 'bgra64le', ['-vf','format=gbrap16le']),
]:
    got = conv(inr, ip, 'rgba64be', extra)
    d = got - want
    nz = int((d != 0).sum())
    print(f"  {name:50s} maxabs={np.abs(d).max():6d}  nonzero={nz:8d}/{d.size}  "
          f"{'EXACT' if nz == 0 else 'lossy'}")

# Characterise the defect for the upstream report: is the error a pure function of value?
got = conv('big_bgra64le.raw', 'bgra64le', 'rgba64be')
err = (got - want)[:, 0]
v   = want[:, 0]
byval = {}
for vv, ee in zip(v[:200000], err[:200000]):
    byval.setdefault(int(vv), set()).add(int(ee))
multi = sum(1 for s in byval.values() if len(s) > 1)
print(f"\n  defect shape: {len(byval)} distinct R values sampled, {multi} of them map to >1 error value")
print(f"                -> error is {'position-dependent (dither-like)' if multi else 'a pure function of the sample'}")
