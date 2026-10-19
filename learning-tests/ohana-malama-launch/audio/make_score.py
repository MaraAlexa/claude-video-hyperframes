"""Original keynote-style score + subtle SFX for the 15s Ohana Mālama launch. Deterministic synthesis, no samples."""
import numpy as np, soundfile as sf, subprocess

SR = 44100
DUR = 15.5
N = int(SR * DUR)
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(21)

def mtof(m): return 440.0 * 2 ** ((m - 69) / 12)
def add(sig, t0, pan=0.0, gain=1.0):
    i = int(t0 * SR); n = min(len(sig), N - i)
    if n <= 0 or i < 0: return
    L[i:i+n] += sig[:n] * gain * (1 - max(pan, 0)); R[i:i+n] += sig[:n] * gain * (1 + min(pan, 0))

def pluck(m, dur, vel):
    f = mtof(m); t = np.arange(int(SR * dur)) / SR; s = np.zeros_like(t)
    for k, a in enumerate([1, .5, .25, .12, .06], 1):
        s += a * np.sin(2 * np.pi * f * k * (1 + 0.0003 * k * k) * t) * np.exp(-t * (2.4 + 1.8 * k))
    return s * np.minimum(t / 0.004, 1) * vel
def bell(m, dur, vel):
    f = mtof(m); t = np.arange(int(SR * dur)) / SR
    s = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * d) for r, a, d in [(1, 1, 1.5), (2.76, .35, 2.6), (5.4, .18, 4.0), (8.93, .08, 6.0)])
    return s * np.minimum(t / 0.002, 1) * vel
def pad(ms, dur, vel, atk=1.0, rel=0.9):
    t = np.arange(int(SR * dur)) / SR; s = np.zeros_like(t)
    for m in ms:
        f = mtof(m); s += np.sin(2 * np.pi * f * t) + .5 * np.sin(2 * np.pi * f * 1.004 * t) + .15 * np.sin(2 * np.pi * f * 2 * t)
    return s * np.minimum(t / atk, 1) * np.minimum((dur - t) / rel, 1) * vel / len(ms)
def sweep_noise(dur, f0, f1, vel, rise=True):
    """filtered-noise whoosh: low-pass cutoff sweeps f0 -> f1; amplitude swells (rise) or decays."""
    n = int(SR * dur); x = rng.standard_normal(n); t = np.linspace(0, 1, n)
    fc = f0 * (f1 / f0) ** t; a = 1 - np.exp(-2 * np.pi * fc / SR)
    y = np.zeros(n); z = 0.0
    for i in range(n): z += a[i] * (x[i] - z); y[i] = z
    hp = y - np.convolve(y, np.ones(400) / 400, mode="same")          # drop the very low rumble
    env = (t ** 2.2) if rise else (1 - t) ** 1.6
    env *= np.minimum(t / 0.05, 1) * np.minimum((1 - t) / 0.08, 1) if rise else 1
    return hp / (np.abs(hp).max() + 1e-9) * env * vel
def boom(dur=1.8, vel=0.9):
    t = np.arange(int(SR * dur)) / SR
    f = 40 + 55 * np.exp(-t * 7); ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 2.3) * vel + sweep_noise(0.5, 3000, 200, 0.25, rise=False)[: len(t)].tolist().__len__() * 0 if False else np.sin(ph) * np.exp(-t * 2.3) * vel
def tick(vel=0.5):
    t = np.arange(int(SR * 0.09)) / SR
    return (np.sin(2 * np.pi * 2600 * t) * np.exp(-t * 90) * .6 + np.sin(2 * np.pi * 420 * t) * np.exp(-t * 55)) * vel

# ---- timeline (seconds) — keep in sync with index.html ----
T_BOOM = 2.45
CARD = [5.55, 7.70, 9.90]
CTA, CLICK, END = 11.7, 13.65, 14.3

# 1. dark opening: low D drone + rising swell into the reveal
add(pad([38, 45], 2.9, 0.5, atk=1.6, rel=0.3), 0.0)
add(sweep_noise(2.1, 180, 7000, 0.45), 0.3)
for k, tt in enumerate([0.9, 1.5, 2.05]):                        # soft heartbeat pulses getting closer
    add(boom(0.5, 0.18 + 0.05 * k), tt)
# 2. the reveal: lights up
add(boom(2.4, 0.9), T_BOOM)
add(sweep_noise(0.7, 600, 9000, 0.35, rise=False), T_BOOM)
for k, m in enumerate([62, 66, 69, 74, 78, 81]):                 # D major add9 shimmer
    add(bell(m + 12, 3.2, 0.10), T_BOOM + 0.05 + k * 0.07, pan=(-1) ** k * 0.35)
# 3. bright harmonic bed after the reveal: D - A - Bm - G - D
chords = [[50, 57, 62, 66], [45, 57, 61, 64], [47, 54, 59, 62], [43, 55, 59, 62], [50, 57, 62, 66]]
for b, ch in enumerate(chords):
    t0 = 2.6 + b * 2.4
    add(pad(ch, 3.4 if b < 4 else 3.2, 0.34), t0)
    for i, off in enumerate([0, 1, 2, 3, 2, 1]):                  # gentle arpeggio
        v = 0.12 if i % 3 == 0 else 0.08
        add(pluck(ch[off] + 12 + (12 if off == 3 else 0), 1.1, v), t0 + 0.1 + i * 0.3, pan=-0.2 + 0.4 * (i % 2))
# 4. the three benefit cards: whoosh in, glass chime, soft tick
for k, tc in enumerate(CARD):
    add(sweep_noise(0.42, 500, 7000, 0.28), tc - 0.4)
    add(bell([81, 85, 88][k], 2.0, 0.22), tc, pan=[-0.3, 0, 0.3][k])
    add(bell([93, 97, 100][k], 1.4, 0.07), tc + 0.02, pan=[-0.3, 0, 0.3][k])
    add(tick(0.28), tc, pan=[-0.3, 0, 0.3][k])
# 5. call to action: transition whoosh, button click, resolved shimmer, tail
add(sweep_noise(0.5, 400, 8000, 0.3), CTA - 0.45)
add(boom(1.4, 0.45), CTA)
add(tick(0.8), CLICK)
for k, m in enumerate([74, 78, 81, 86, 90]):
    add(bell(m, 2.6, 0.12), CLICK + 0.1 + k * 0.09, pan=(-1) ** k * 0.3)
add(pad([50, 57, 62, 66, 74], 2.2, 0.30, atk=0.4, rel=1.2), 13.3)

# ---- light room reverb on everything, gentle master ----
def conv(a, b):
    n = len(a) + len(b) - 1; nf = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, nf) * np.fft.rfft(b, nf), nf)[:len(a)]
tt = np.arange(int(SR * 1.8)) / SR
ir = rng.standard_normal(len(tt)) * np.exp(-tt * 2.8)
for i in range(1, len(ir)): ir[i] = 0.6 * ir[i] + 0.4 * ir[i - 1]
ir /= np.sqrt((ir ** 2).sum()) * 12
out = np.stack([L * 0.8 + conv(L, ir) * 0.25 * 6, R * 0.8 + conv(R, ir[::-1].copy()) * 0.25 * 6], 1)
fade = np.ones(N); fi = int(0.05 * SR); fo = int(1.3 * SR)
fade[:fi] = np.linspace(0, 1, fi); fade[-fo:] = np.linspace(1, 0, fo)
out *= fade[:, None]
out /= np.abs(out).max() / 0.9
sf.write("score.wav", out, SR)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", "score.wav", "-b:a", "192k", "score.mp3"], check=True)
print("ok", round(len(out) / SR, 2))
