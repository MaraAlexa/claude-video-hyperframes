"""Spanish voiceover (Kokoro ef_dora, local) -> clean keynote voice: warm, even level, hint of room."""
import subprocess, json, numpy as np, soundfile as sf

LINES = {
 "v1": "Presentamos Ohana Malama.",
 "v2": "La academia más cercana. La más humana.",
 "v3": "Clases en directo y diferido.",
 "v4": "Dudas y supuestos, en directo.",
 "v5": "Seminarios en directo.",
 "v7": "Plazas limitadas. Únete a la familia Malama.",
}
SR = 24000
rng = np.random.default_rng(3)
t = np.arange(int(SR * 1.2)) / SR
ir = rng.standard_normal(len(t)) * np.exp(-t * 4.5)
ir = np.concatenate([np.zeros(int(.015 * SR)), ir])
for i in range(1, len(ir)): ir[i] = 0.55 * ir[i] + 0.45 * ir[i - 1]
ir /= np.sqrt((ir ** 2).sum())
def conv(a, b):
    n = len(a) + len(b) - 1; nf = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, nf) * np.fft.rfft(b, nf), nf)[:n]

durs = {}
for k, txt in LINES.items():
    subprocess.run(["npx", "hyperframes", "tts", txt, "-v", "ef_dora", "-s", "1.0", "-l", "es", "-o", f"raw_{k}.wav", "--json"], capture_output=True, text=True, check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", f"raw_{k}.wav", "-af",
        "highpass=f=80,bass=g=1.5:f=180,treble=g=-1:f=5000,"
        "acompressor=threshold=-24dB:ratio=2.2:attack=15:release=200:makeup=2",
        "-ar", str(SR), "-ac", "1", f"eq_{k}.wav"], check=True)
    x, _ = sf.read(f"eq_{k}.wav")
    dry = np.concatenate([x, np.zeros(len(ir))])
    wet = conv(x, ir); wet = np.pad(wet, (0, max(0, len(dry) - len(wet))))[: len(dry)]
    y = dry * 0.95 + wet * 0.14
    y[: int(.015 * SR)] *= np.linspace(0, 1, int(.015 * SR)); n = int(.25 * SR); y[-n:] *= np.linspace(1, 0, n)
    y *= 0.125 / (np.sqrt((y[: len(x)] ** 2).mean()) + 1e-9)
    y = np.clip(y, -0.95, 0.95)
    sf.write(f"vo_{k}.wav", y, SR)
    durs[k] = round(len(x) / SR, 2)
print(json.dumps(durs))
