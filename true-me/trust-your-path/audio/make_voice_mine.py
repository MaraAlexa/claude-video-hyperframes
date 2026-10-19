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

def real_edges(a, b, prev_end=None, next_start=None):
    """Quiet consonants (soft final 's', 't', 'th') fall under the silence threshold; grow the phrase until the true floor."""
    win = int(.03 * SR)
    def level(t): 
        i = int(t * SR); f = x[i:i + win]
        return 20 * np.log10(np.sqrt((f ** 2).mean()) + 1e-9) if len(f) else -120
    FLOOR = -76
    lo = prev_end + 0.12 if prev_end is not None else 0.0
    hi = next_start - 0.12 if next_start is not None else len(x) / SR
    a2 = a
    while a2 - 0.01 > lo and a - a2 < 0.25 and level(a2 - 0.03) > FLOOR: a2 -= 0.01
    b2 = b
    quiet = 0
    while b2 + 0.01 < hi and b2 - b < 0.55:
        b2 += 0.01
        quiet = quiet + 1 if level(b2) < FLOOR else 0
        if quiet >= 3: b2 -= 0.03; break
    return round(a2, 2), round(b2, 2)

allsegs = [sg for segs in LINES.values() for sg in segs]
ext = {}
for i, (a, b) in enumerate(allsegs):
    ext[(a, b)] = real_edges(a, b, allsegs[i - 1][1] if i else None, allsegs[i + 1][0] if i + 1 < len(allsegs) else None)
ORIG_END = {ext[sg]: sg[1] for sg in allsegs}
LINES = {k: [ext[sg] for sg in segs] for k, segs in LINES.items()}
print({k: v for k, v in LINES.items()}, file=__import__("sys").stderr)

WORDS = {"t1": [8], "m1": [4, 3], "m2": [3, 3], "m3": [5], "m4": [4], "t3": [4, 4], "t4": [3, 4]}
PAD_IN, PAD_OUT, PAUSE_CAP = 0.06, 0.04, 0.3

rng = np.random.default_rng(5)
t = np.arange(int(SR * 1.5)) / SR
ir = rng.standard_normal(len(t)) * np.exp(-t * 3.4)
ir = np.concatenate([np.zeros(int(.022 * SR)), ir])
for i in range(1, len(ir)): ir[i] = 0.55 * ir[i] + 0.45 * ir[i - 1]   # soften highs
ir /= np.sqrt((ir ** 2).sum())
def conv(a, b):
    n = len(a) + len(b) - 1; nf = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, nf) * np.fft.rfft(b, nf), nf)[:n]

meta = {}
for k, segs in LINES.items():
    parts, rel, pos = [], [], 0.0
    for i, (a, b) in enumerate(segs):
        lead = PAD_IN if i == 0 else 0.0
        seg = x[int((a - (PAD_IN if i == 0 else 0.05)) * SR): int((b + PAD_OUT) * SR)].copy()
        # lift the quiet word-ending (soft 's', 't', 'th') that sat below the detection threshold by ~10 dB
        t_start = a - (PAD_IN if i == 0 else 0.05)
        k0 = int((ORIG_END[(a, b)] - 0.03 - t_start) * SR)
        if 0 < k0 < len(seg):
            g = np.ones(len(seg)); ramp = int(.04 * SR)
            g[k0:] = 10 ** (10 / 20); g[k0:k0 + ramp] = np.linspace(1, 10 ** (10 / 20), min(ramp, len(seg) - k0))
            seg *= g
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
        "highpass=f=90,equalizer=f=300:t=q:w=1:g=-4,equalizer=f=1000:t=q:w=1.2:g=-2.5,"
        "equalizer=f=2200:t=q:w=0.9:g=9,equalizer=f=3500:t=q:w=1:g=3,"
        "aexciter=level_in=1:level_out=1:amount=0.8:drive=5:blend=0:freq=3500:ceil=10000,lowpass=f=8500,"
        "acompressor=threshold=-30dB:ratio=2:attack=15:release=200:makeup=1.5",
        "/tmp/claude-0/seg_eq.wav"], check=True)
    y, _ = sf.read("/tmp/claude-0/seg_eq.wav")
    rms = np.sqrt((y ** 2).mean()); y = y * (0.115 / (rms + 1e-9))
    dry = np.concatenate([y, np.zeros(len(ir))])
    wet = conv(y, ir); wet = np.pad(wet, (0, max(0, len(dry) - len(wet))))[: len(dry)]
    out = dry * 0.9 + wet * 0.2
    n = int(.3 * SR); out[-n:] *= np.linspace(1, 0, n)
    out = np.clip(out, -0.95, 0.95)
    sf.write(f"me_{k}.wav", out, SR)
    meta[k] = {"d": round(d, 2), "parts": [[WORDS[k][i], rel[i][0], rel[i][1]] for i in range(len(rel))]}
print(json.dumps(meta))
