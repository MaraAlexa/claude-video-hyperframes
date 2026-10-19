"""Edit Mara's recorded take (my_voice_raw.m4a) into per-line clips: cut phrases, cap pauses, level, warm EQ, soft room."""
import subprocess, json, numpy as np, soundfile as sf

SR = 44100
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", "my_voice_raw.m4a", "-ac", "1", "-ar", str(SR), "/tmp/claude-0/take.wav"], check=True)
x, _ = sf.read("/tmp/claude-0/take.wav")

# phrase boundaries found by silence detection (seconds in the take), grouped by script line
LINES = {
 "t1": [(3.09, 5.26)],
 "m1": [(6.54, 7.72), (8.67, 9.18)],
 "m2": [(10.10, 11.28), (11.72, 12.46)],
 "m3": [(13.44, 15.06)],
 "m4": [(15.90, 17.26)],
 "t3": [(18.56, 20.25), (20.82, 21.58)],
 "t4": [(22.43, 23.03), (24.17, 25.09)],
}
WORDS = {"t1": [8], "m1": [4, 3], "m2": [3, 3], "m3": [5], "m4": [4], "t3": [4, 4], "t4": [3, 4]}
PAD_IN, PAD_OUT, PAUSE_CAP = 0.08, 0.12, 0.38

rng = np.random.default_rng(5)
t = np.arange(int(SR * 1.5)) / SR
ir = rng.standard_normal(len(t)) * np.exp(-t * 3.4)
ir = np.concatenate([np.zeros(int(.022 * SR)), ir]); ir /= np.sqrt((ir ** 2).sum())
def conv(a, b):
    n = len(a) + len(b) - 1; nf = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, nf) * np.fft.rfft(b, nf), nf)[:n]

meta = {}
for k, segs in LINES.items():
    parts, rel, pos = [], [], 0.0
    for i, (a, b) in enumerate(segs):
        lead = PAD_IN if i == 0 else 0.0
        seg = x[int((a - (PAD_IN if i == 0 else 0.05)) * SR): int((b + PAD_OUT) * SR)].copy()
        f = int(.015 * SR); seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
        if i > 0:
            gap = min(segs[i][0] - segs[i - 1][1] - PAD_OUT, PAUSE_CAP)
            parts.append(np.zeros(int(max(gap, 0.15) * SR))); pos += max(gap, 0.15)
        rel.append([round(pos + (PAD_IN if i == 0 else 0.05), 2), round(b - a, 2)])
        parts.append(seg); pos += len(seg) / SR
    y = np.concatenate(parts)
    d = len(y) / SR
    open("/tmp/claude-0/seg.wav", "wb").close()
    sf.write("/tmp/claude-0/seg_in.wav", y, SR)
    # level up (take is very quiet), warm, gentle compression
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", "/tmp/claude-0/seg_in.wav", "-af",
        "highpass=f=80,bass=g=2:f=190,treble=g=-2:f=4500,"
        "acompressor=threshold=-34dB:ratio=2.6:attack=15:release=200:makeup=2",
        "/tmp/claude-0/seg_eq.wav"], check=True)
    y, _ = sf.read("/tmp/claude-0/seg_eq.wav")
    rms = np.sqrt((y ** 2).mean()); y = y * (0.115 / (rms + 1e-9))
    dry = np.concatenate([y, np.zeros(len(ir))])
    wet = conv(y, ir); wet = np.pad(wet, (0, max(0, len(dry) - len(wet))))[: len(dry)]
    out = dry * 0.86 + wet * 0.28
    n = int(.3 * SR); out[-n:] *= np.linspace(1, 0, n)
    out = np.clip(out, -0.95, 0.95)
    sf.write(f"me_{k}.wav", out, SR)
    meta[k] = {"d": round(d, 2), "parts": [[WORDS[k][i], rel[i][0], rel[i][1]] for i in range(len(rel))]}
print(json.dumps(meta))
