"""Soft, slow voiceover: Kokoro (local) -> warm EQ + gentle compression -> small-room reverb."""
import subprocess, json, numpy as np, soundfile as sf, sys

VOICE, SPEED = "af_sky", "0.9"
LINES = {
 "t1": "You don't have to see... the whole path.",
 "m1": "It's not about knowing... where it leads.",
 "m2": "It's about trusting... the next step.",
 "m3": "It's not about being certain.",
 "m4": "It's about being willing.",
 "t3": "Stepping stones never ask... to see the shore.",
 "t4": "You're not lost... You're on your way.",
}
rng = np.random.default_rng(11)
SR = 24000

def conv(x, ir):
    n = len(x) + len(ir) - 1; nf = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, nf) * np.fft.rfft(ir, nf), nf)[:n]

t = np.arange(int(SR * 1.5)) / SR
ir = rng.standard_normal(len(t)) * np.exp(-t * 3.4)
ir = np.concatenate([np.zeros(int(.022 * SR)), ir]); ir /= np.sqrt((ir ** 2).sum())

durs = {}
for k, txt in LINES.items():
    raw = f"raw_{k}.wav"
    r = subprocess.run(["npx", "hyperframes", "tts", txt, "-v", VOICE, "-s", SPEED, "-o", raw, "--json"], capture_output=True, text=True)
    assert json.loads(r.stdout.strip().splitlines()[-1])["ok"], r.stdout
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-af",
        "highpass=f=85,bass=g=2.5:f=190,treble=g=-3.5:f=4200,lowpass=f=9000,"
        "acompressor=threshold=-24dB:ratio=2.4:attack=20:release=220:makeup=3",
        "-ar", str(SR), "-ac", "1", f"eq_{k}.wav"], check=True)
    x, _ = sf.read(f"eq_{k}.wav")
    dry = np.concatenate([x, np.zeros(len(ir))])
    wet = conv(x, ir)[: len(dry)] if len(x) else dry
    y = dry[: len(wet)] * 0.84 + wet * 0.30
    # soft fade in/out so there are no clicks
    y[: int(.02 * SR)] *= np.linspace(0, 1, int(.02 * SR)); n = int(.25 * SR); y[-n:] *= np.linspace(1, 0, n)
    y *= 0.16 / (np.sqrt((y[: len(x)] ** 2).mean()) + 1e-9) * 0.75
    y = np.clip(y, -0.95, 0.95)
    sf.write(f"vo_{k}.wav", y, SR)
    durs[k] = round(len(x) / SR, 2)
print(json.dumps(durs))
