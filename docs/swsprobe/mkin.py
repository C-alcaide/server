import numpy as np
W, H = 16, 16
n = W*H
rng = np.random.RandomState(7)
# Deterministic, wide coverage: rails, 8-bit-replicated values, and random 16-bit.
vals = []
for i in range(n):
    if i < 4:
        vals.append([0, 0, 0, 65535][i:i+1]*4 if False else [[0,0,0,65535],[65535,65535,65535,65535],[0,65535,0,65535],[65535,0,0,65535]][i])
    elif i < 40:
        b = (i*7) % 256
        vals.append([b*257, ((i*13)%256)*257, ((i*29)%256)*257, 65535])
    else:
        vals.append([int(x) for x in rng.randint(0, 65536, 4)])
a = np.array(vals, dtype='<u2')          # per pixel: [B,G,R,A] in bgra64le order
a[:, 3] = 65535                          # opaque: alpha games are not what we are testing
a.tofile('in_bgra64le.raw')
# The same picture written as rgba64le, for the control arm.
b = a.copy()
b[:, [0, 2]] = a[:, [2, 0]]
b.tofile('in_rgba64le.raw')
print("wrote", n, "px")
