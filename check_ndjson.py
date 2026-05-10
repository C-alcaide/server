import json

with open("ISJ_Show-Iluminacion-Universo_1-v4_20251119.ndjson") as f:
    lines = [f.readline() for _ in range(12)]

h = json.loads(lines[0])
print("HEADER:", h.get("copyright"), h.get("version"), len(h.get("channels",[])), "channels")
print()

prev_dt = 0
for i, line in enumerate(lines[1:11], 1):
    obj = json.loads(line)
    dt = obj["dt"]
    delta = dt - prev_dt
    active = sum(1 for v in obj["data"] if v > 0)
    print(f"  Frame {i:2d}: dt={dt:>12d} us  ({dt/1e6:>9.3f}s)  delta={delta:>8d} us ({delta/1e3:>7.1f}ms)  active_ch={active}")
    prev_dt = dt

# Also show some stats
print()
all_frames = []
with open("ISJ_Show-Iluminacion-Universo_1-v4_20251119.ndjson") as f:
    for idx, line in enumerate(f):
        if idx == 0:
            continue
        obj = json.loads(line)
        all_frames.append(obj["dt"])

deltas = [all_frames[i] - all_frames[i-1] for i in range(1, len(all_frames))]
print(f"Total frames: {len(all_frames)}")
print(f"Duration: {(all_frames[-1] - all_frames[0])/1e6:.3f}s")
print(f"Delta min: {min(deltas)/1e3:.1f}ms  max: {max(deltas)/1e3:.1f}ms  avg: {sum(deltas)/len(deltas)/1e3:.1f}ms")
print(f"Avg rate: {1e6 / (sum(deltas)/len(deltas)):.1f} Hz")
