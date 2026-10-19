"""Original bright, hopeful instrumental (G major, 96 BPM, ~20s). Deterministic; no samples used."""
import numpy as np, soundfile as sf, subprocess

SR = 44100
BPM = 96
BEAT = 60 / BPM
BARS = 8
N = int(SR * BEAT * 4 * BARS) + SR * 3          # tail for reverb
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(7)

def mtof(m): return 440.0 * 2 ** ((m - 69) / 12)

def add(sig, t0, pan=0.0):
    i = int(t0 * SR)
    n = min(len(sig), N - i)
    if n <= 0: return
    L[i:i+n] += sig[:n] * (1 - max(pan, 0)); R[i:i+n] += sig[:n] * (1 + min(pan, 0))

def pluck(m, dur, vel):                          # soft piano-like tone
    f = mtof(m); t = np.arange(int(SR * dur)) / SR
    s = np.zeros_like(t)
    for k, a in enumerate([1, .55, .28, .14, .07], 1):
        s += a * np.sin(2 * np.pi * f * k * (1 + 0.0003 * k * k) * t) * np.exp(-t * (2.2 + 1.8 * k))
    s *= np.minimum(t / 0.004, 1)
    return s * vel

def bell(m, dur, vel):                           # glockenspiel-like sparkle
    f = mtof(m); t = np.arange(int(SR * dur)) / SR
    s = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * d)
            for r, a, d in [(1, 1, 1.6), (2.76, .35, 2.6), (5.4, .18, 4.0), (8.93, .08, 6.0)])
    return s * np.minimum(t / 0.002, 1) * vel

def pad(ms, dur, vel):                           # warm sustained chord
    t = np.arange(int(SR * dur)) / SR
    s = np.zeros_like(t)
    for m in ms:
        f = mtof(m)
        s += np.sin(2 * np.pi * f * t) + .5 * np.sin(2 * np.pi * f * 1.004 * t) + .15 * np.sin(2 * np.pi * f * 2 * t)
    env = np.minimum(t / 1.1, 1) * np.minimum((dur - t) / 0.9, 1)
    return s * env * vel / len(ms)

# G  D/F#  Em  C | G  D  C  G
chords = [[55, 59, 62, 67], [54, 57, 62, 66], [52, 55, 59, 64], [48, 52, 55, 60],
          [55, 59, 62, 67], [50, 54, 57, 62], [48, 52, 55, 60], [55, 59, 62, 67]]
arp = [0, 1, 2, 3, 2, 1, 2, 1]                   # eighth-note figure over chord tones

for b, ch in enumerate(chords):
    t0 = b * BEAT * 4
    grow = 0.55 + 0.45 * min(b / 4, 1)           # builds toward the middle
    add(pad([n - 12 for n in ch[:3]] + [ch[0] - 12], BEAT * 4 + .5, 0.20 * grow), t0)
    for i, a in enumerate(arp):
        v = (0.22 if i % 4 == 0 else 0.15) * grow * (1 + 0.1 * rng.standard_normal())
        add(pluck(ch[a] + 12 * (a == 3 and b < 2 or 0) + 12, 1.2, v), t0 + i * BEAT / 2, pan=-0.25 + 0.5 * (i % 2))
    add(pluck(ch[0] - 12, 2.0, 0.28 * grow), t0)  # bass note on the downbeat

# bell melody enters mid-way and blooms in the finale (G A B D E pentatonic)
mel = {  # bar: [(beat, midi, beats)]
    3: [(0, 74, 1.5), (1.5, 76, .5), (2, 79, 2)],
    4: [(0, 83, 1), (1, 81, 1), (2, 79, 2)],
    5: [(0, 78, 1), (1, 81, 1), (2, 86, 2)],
    6: [(0, 84, 1), (1, 83, 1), (2, 79, 1), (3, 76, 1)],
    7: [(0, 91, 4)],
}
for bar, notes in mel.items():
    for beat, m, d in notes:
        t = bar * BEAT * 4 + beat * BEAT
        v = 0.17 if bar < 6 else 0.22
        add(bell(m, min(3.5, d * BEAT + 1.8), v), t, pan=0.2)
        if bar >= 5: add(bell(m + 12, 2.0, v * 0.45), t + 0.01, pan=-0.3)
# shimmer on the final chord
for k, m in enumerate([86, 91, 95, 98, 103]):
    add(bell(m, 3.0, 0.07), 7 * BEAT * 4 + 0.3 + k * 0.28, pan=(-1) ** k * 0.4)

# simple stereo reverb (synthetic exponential noise impulse), 28% wet
def conv(x, ir):
    n = len(x) + len(ir) - 1; nf = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, nf) * np.fft.rfft(ir, nf), nf)[:len(x)]
ir_t = np.arange(int(SR * 2.2)) / SR
irL = rng.standard_normal(len(ir_t)) * np.exp(-ir_t * 2.6); irR = rng.standard_normal(len(ir_t)) * np.exp(-ir_t * 2.6)
irL /= np.abs(irL).sum() ** .5 * 14; irR /= np.abs(irR).sum() ** .5 * 14
wL, wR = conv(L, irL), conv(R, irR)
out = np.stack([L * 0.72 + wL * 0.28 * 6, R * 0.72 + wR * 0.28 * 6], 1)
out[: int(0.05 * SR)] *= np.linspace(0, 1, int(0.05 * SR))[:, None]
out = out[: int(SR * 20.5)]
out /= np.abs(out).max() / 0.89
sf.write("music_bright.wav", out, SR)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", "music_bright.wav", "-b:a", "192k", "music_bright.mp3"], check=True)
print("ok", len(out) / SR)
