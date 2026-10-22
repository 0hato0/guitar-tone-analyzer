"""動作確認用の合成曲を作る（答えが分かっている試験データ）"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

OUT = Path(sys.argv[1])
SR = 48000
BPM = 175.0
Q = 60 / BPM
DUR = 56.0
rng = np.random.default_rng(1)
n = int(DUR * SR)
t_all = np.arange(n) / SR


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def tone(midis, dur, amp=0.3, vib=None):
    tt = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(tt)
    for m in midis:
        f = hz(m)
        ph = 2 * np.pi * f * tt
        if vib:
            rate, cents = vib
            ph = 2 * np.pi * np.cumsum(f * 2 ** (cents / 1200 * np.sin(2 * np.pi * rate * tt))) / SR
        for h in range(1, 12):
            if f * h < SR / 2.2:
                y += np.sin(h * ph) / h
    y = np.tanh(3 * y / max(1, len(midis))) * amp
    env = np.minimum(1, tt / 0.004) * np.exp(-tt * 1.5)
    rel = int(0.008 * SR)
    env[-rel:] *= np.linspace(1, 0, rel)
    return y * env


def place(buf, y, t):
    a = int(t * SR)
    b = min(len(buf), a + len(y))
    buf[a:b] += y[:b - a]


gL = np.zeros(n)
gR = np.zeros(n)
riffR = np.zeros(n)
chordsL = np.zeros(n)
E5 = [40, 47, 52]
A5 = [45, 52, 57]
D5 = [50, 57, 62]
C5 = [48, 55, 60]
prog = [E5, C5, D5, A5]

g0 = 8.0
# 8〜19秒：ギター1本、2拍ごとに短く切るコード（乾いた音）
t = g0
k = 0
while t < 19.0 - 1e-6:
    y = tone(prog[(k // 2) % 4], Q * 0.55, 0.35)
    place(gL, y * 10 ** (-4 / 20), t)
    place(gR, y, t)
    t += 2 * Q
    k += 1
# 19〜33秒：右＝単音リフ（8分、休符あり）、左＝コード（4分）
riff = [64, 67, 69, None, 71, 69, None, 67]
t = g0 + 2 * Q * k
k = 0
while t < 33.0 - 1e-6:
    m = riff[k % len(riff)]
    if m is not None:
        place(riffR, tone([m], Q * 0.5 * 0.7, 0.3), t)
    if k % 2 == 0:
        place(chordsL, tone(prog[(k // 8) % 4], Q * 0.9, 0.25), t)
    t += Q / 2
    k += 1
t_sec3 = t
# ディレイ 290ms fb 0.3、ミックス -6dB（右だけ）
D = int(0.290 * SR)
dl = np.zeros(n)
x = riffR.copy()
g = 10 ** (-6 / 20)
for rep in range(1, 7):
    sh = np.zeros(n)
    sh[D * rep:] = x[:n - D * rep]
    dl += sh * g * 0.3 ** (rep - 1)
gR += riffR + dl
gL += chordsL

# 33〜46秒：左＝フェイザー付きのコード、右＝伸ばした単音（コーラスっぽい音高の揺れ）
phL = np.zeros(n)
t = t_sec3
k = 0
while t < 46.0 - 1e-6:
    place(phL, tone(prog[(k // 2) % 4], Q * 1.9, 0.25), t)
    if k % 4 == 0:
        place(gR, tone([71 if (k // 4) % 2 == 0 else 69], Q * 3.8, 0.3, vib=(1.3, 8)), t)
    t += 2 * Q
    k += 1
t_kime = t
# フェイザー：4段のオールパスを周期2.74秒で動かす
a_ph = int(t_sec3 * SR)
b_ph = int(t_kime * SR)
seg = phL[a_ph:b_ph]
tt = np.arange(len(seg)) / SR
fc = 500 * 2 ** (3 * (0.5 - 0.5 * np.cos(2 * np.pi * tt / 2.74)))
out = seg.copy()
blk = 256
state = [np.zeros(1) for _ in range(4)]
y_ap = np.zeros_like(seg)
for i0 in range(0, len(seg), blk):
    f = fc[i0]
    c = (np.tan(np.pi * f / SR) - 1) / (np.tan(np.pi * f / SR) + 1)
    z = seg[i0:i0 + blk]
    for s in range(4):
        z, state[s] = signal.lfilter([c, 1], [1, c], z, zi=state[s])
    y_ap[i0:i0 + blk] = z
gL[a_ph:b_ph] += 0.5 * (seg + y_ap)

# キメ：t_kime から1.4秒全員停止、そのあと最後まで2拍ごとのコード（両方）
t = t_kime + 1.4
k = 0
while t < DUR - 1:
    y = tone(prog[(k // 2) % 4], Q * 1.8, 0.3)
    place(gL, y, t)
    place(gR, y * 0.9, t)
    t += 2 * Q
    k += 1

# 他の楽器：0〜8秒はノイズの効果音、それ以降はキック・スネア・ベース（キメの間は止める）
oth = np.zeros((2, n))
noise = rng.normal(0, 0.05, int(g0 * SR)) * np.linspace(0.2, 1, int(g0 * SR))
oth[:, :len(noise)] += noise
beat_t = np.arange(g0, DUR - 1, Q)
for i, bt in enumerate(beat_t):
    if t_kime - 0.01 <= bt < t_kime + 1.4:
        continue
    tt = np.arange(int(0.12 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 40)) * tt) * np.exp(-tt * 25) * 0.5
    if i % 2 == 1:
        kick = rng.normal(0, 0.25, len(tt)) * np.exp(-tt * 30)
    a = int(bt * SR)
    oth[:, a:a + len(kick)] += kick
    bass = tone([prog[(i // 4) % 4][0] - 12], Q * 0.9, 0.25)
    oth[:, a:a + len(bass)] += bass[:min(len(bass), n - a)]

gtr = np.vstack([gL, gR])
mix = gtr + oth
peak = np.max(np.abs(mix)) * 1.05
mix /= peak
gtr /= peak

(OUT / "audio").mkdir(parents=True, exist_ok=True)
sf.write(OUT / "audio" / "原曲.wav", mix.T, SR)
# ギター分離音源：44.1kHz、25ms 前にずらす（先頭を削る）
from scipy.signal import resample_poly
g44 = resample_poly(gtr, 147, 160, axis=1)
cut = int(0.025 * 44100)
g44 = g44[:, cut:]
sf.write(OUT / "audio" / "1_テスト曲_(Guitar).mp3", g44.T, 44100)
(OUT / "request.md").write_text("# 曲ごとの情報\n- 曲名／アーティスト：テスト曲\n", encoding="utf-8")
print("sections:", g0, 19.0, t_sec3, t_kime, t_kime + 1.4)
