"""modulation（揺れ系の判定材料）

- 倍音の細かい山をならした対数周波数のスペクトログラム（250Hz〜8kHz）から、周波数ごとの時間平均と
  各瞬間の平均を引いた画像（赤＝山、青＝谷）を、パートごと・左右別に保存する
- 同じ画像を線形周波数（100Hz〜5kHz）でも保存する（フェイザー＝谷が少数、フランジャー＝等間隔の谷が多数）
- 谷の位置の時間変化の自己相関から、揺れの周期の候補を出す。その区間のコードが変わる周期も出し、
  同じなら注意を付ける
- 伸ばした単音で、音高の細かい揺れ（セント）の周期と深さ（コーラスの手がかり）
- 伸ばした単音で、音量の周期的な上下（トレモロの手がかり）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import librosa
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from . import dsp, plotting
from .common import CH_JA, ENV_HOP, HOP, fmt_time, nan_median, pow_db, save_csv, save_json

N_FFT = 4096
MOD_HOP = 1024
LOG_FMIN, LOG_FMAX, LOG_BPO = 250.0, 8000.0, 24
LIN_FMIN, LIN_FMAX = 100.0, 5000.0
VALLEY_FMIN, VALLEY_FMAX = 400.0, 5000.0
PERIOD_MIN, PERIOD_MAX = 0.5, 8.0
MAX_REGIONS = 8

REFERENCE = {
    "image": "赤＝その周波数がいつもより強い、青＝弱い（谷）。谷が斜めに動く弧が周期的に続けば揺れ系の手がかり",
    "valley_period": "谷の位置の自己相関から出した周期の候補。コード進行の周期と同じなら、コードの変化を見ている可能性がある",
    "notch_count": "線形周波数の画像で1フレームに見える谷の数の中央値。少数ならフェイザー、等間隔に多数ならフランジャーの目安",
    "vibrato": "伸ばした単音の音高の揺れ。コーラスなら数セント〜十数セントが周期的に揺れる（指のビブラートとも区別が必要）",
    "tremolo": "伸ばした単音の音量の周期的な上下",
}


def _log_weights(freqs):
    n = int(np.floor(LOG_BPO * np.log2(LOG_FMAX / LOG_FMIN))) + 1
    centers = LOG_FMIN * 2 ** (np.arange(n) / LOG_BPO)
    W = np.zeros((n, len(freqs)), np.float32)
    df = freqs[1] - freqs[0]
    for i, fc in enumerate(centers):
        lo, hi = fc * 2 ** (-0.5 / LOG_BPO), fc * 2 ** (0.5 / LOG_BPO)
        sel = np.flatnonzero((freqs >= lo) & (freqs < hi))
        if len(sel) >= 2:
            W[i, sel] = 1.0 / len(sel)
        else:
            k = int(np.floor(fc / df))
            frac = fc / df - k
            W[i, k] = 1 - frac
            W[i, k + 1] = frac
    return centers, W


class ModSpec:
    """1チャンネル分の、ならしたスペクトログラム（dB）"""

    def __init__(self, ctx, kind, ch, t0=0.0, t1=None):
        """t0〜t1 を渡すとその範囲だけ計算する（--check 用）"""
        y = ctx.sig(kind, ch)
        a = int(max(t0, 0) * ctx.sr)
        b = len(y) if t1 is None else min(len(y), int(t1 * ctx.sr))
        self.t_off = a / ctx.sr
        P = dsp.stft_power(y[a:b], N_FFT, MOD_HOP)
        freqs = librosa.fft_frequencies(sr=ctx.sr, n_fft=N_FFT)
        self.centers, W = _log_weights(freqs)
        self.log = gaussian_filter(pow_db(W @ P), sigma=(2.0, 1.5)).astype(np.float32)
        sel = (freqs >= LIN_FMIN) & (freqs <= LIN_FMAX)
        self.lin_freqs = freqs[sel]
        self.lin = gaussian_filter(pow_db(P[sel]), sigma=(3.0, 1.5)).astype(np.float32)
        del P
        self.hop_s = MOD_HOP / ctx.sr
        env = ctx.env(kind, ch)
        n = self.log.shape[1]
        idx = np.clip(np.round((self.t_off + np.arange(n) * self.hop_s) / ctx.env_hop_s).astype(int), 0, len(env) - 1)
        self.active = env[idx] > ctx.active_threshold(kind)

    def frames(self, t0, t1):
        return (max(0, int(round((t0 - self.t_off) / self.hop_s))),
                min(self.log.shape[1], int(round((t1 - self.t_off) / self.hop_s))))

    def time_of(self, i):
        return self.t_off + i * self.hop_s

    @staticmethod
    def normalize(Z, act):
        Z = Z.astype(np.float64).copy()
        if act.sum() < 3:
            return np.full_like(Z, np.nan)
        Z -= Z[:, act].mean(axis=1, keepdims=True)
        Z -= Z.mean(axis=0, keepdims=True)
        Z[:, ~act] = np.nan
        return Z

    def section(self, t0, t1):
        a, b = self.frames(t0, t1)
        act = self.active[a:b]
        return a, b, act, self.normalize(self.log[:, a:b], act), self.normalize(self.lin[:, a:b], act)


def valley_track(Zlog, centers, act):
    sel = (centers >= VALLEY_FMIN) & (centers <= VALLEY_FMAX)
    Z = gaussian_filter1d(np.nan_to_num(Zlog[sel]), sigma=3, axis=0)
    k = np.argmin(Z, axis=0)
    f = centers[sel][k]
    depth = Z[k, np.arange(Z.shape[1])]
    f = np.where(act, f, np.nan)
    depth = np.where(act, depth, np.nan)
    return f, depth


def notch_stats(Zlin, freqs, act):
    counts, cvs = [], []
    df = freqs[1] - freqs[0]
    for i in np.flatnonzero(act):
        col = Zlin[:, i]
        if not np.all(np.isfinite(col)):
            continue
        pk, _ = signal.find_peaks(-col, prominence=3.0, distance=max(1, int(60 / df)))
        pk = pk[(freqs[pk] >= 200)]
        counts.append(len(pk))
        if len(pk) >= 4:
            d = np.diff(freqs[pk])
            cvs.append(float(np.std(d) / (np.mean(d) + 1e-9)))
    return (float(np.median(counts)) if counts else np.nan,
            float(np.median(cvs)) if cvs else np.nan)


def _period_peaks(x, hop_s, max_len_s):
    if np.sum(np.isfinite(x)) < 20:
        return []
    r = dsp.acf(x)
    lags = np.arange(len(r)) * hop_s
    return dsp.top_peaks(lags, r, n=3, min_x=PERIOD_MIN, max_x=min(PERIOD_MAX, max_len_s / 2),
                         prominence=0.05)


def row_acf_peaks(Z, hop_s):
    if Z.shape[1] < 20 or not np.any(np.isfinite(Z)):
        return []
    acc = None
    for row in Z:
        r = dsp.acf(row)
        acc = r if acc is None else acc + r
    acc = acc / Z.shape[0]
    lags = np.arange(len(acc)) * hop_s
    return dsp.top_peaks(lags, acc, n=3, min_x=PERIOD_MIN, max_x=min(PERIOD_MAX, Z.shape[1] * hop_s / 2),
                         prominence=0.02)


def chord_period(ctx, ch, t0, t1):
    C = ctx.chroma("gtr", ch)[:, ctx.frame_index(t0):ctx.frame_index(t1)].astype(np.float64)
    act = ctx.active_frames("gtr", ch)[ctx.frame_index(t0):ctx.frame_index(t1)]
    if C.shape[1] < 50 or act.mean() < 0.2:
        return []
    C = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)
    C[:, ~act[:C.shape[1]]] = np.nan
    acc = None
    for row in C:
        r = dsp.acf(row)
        acc = r if acc is None else acc + r
    acc /= 12
    lags = np.arange(len(acc)) * ctx.hop_s
    return dsp.top_peaks(lags, acc, n=3, min_x=PERIOD_MIN, max_x=min(PERIOD_MAX, (t1 - t0) / 2), prominence=0.03)


# ---- 伸ばした単音の揺れ ---------------------------------------------------
def sustained_regions(ctx, ch, t0, t1, min_len=0.35):
    count, lowest = ctx.notes("gtr", ch)
    on = set(ctx.onsets("gtr", ch).tolist())
    a, b = ctx.frame_index(t0), min(ctx.frame_index(t1), len(count))
    regs = []
    i = a
    while i < b:
        if count[i] == 1 and np.isfinite(lowest[i]):
            j = i + 1
            while j < b and count[j] == 1 and lowest[j] == lowest[i] and j not in on:
                j += 1
            if (j - i) * ctx.hop_s >= min_len:
                regs.append((i * ctx.hop_s, j * ctx.hop_s, float(lowest[i])))
            i = j
        else:
            i += 1
    regs.sort(key=lambda r: -(r[1] - r[0]))
    return regs[:MAX_REGIONS]


def _periodicity(x, hop_s, lo, hi):
    x = np.asarray(x, float)
    if np.sum(np.isfinite(x)) < 10:
        return np.nan, np.nan
    r = dsp.acf(x)
    lags = np.arange(len(r)) * hop_s
    pk = dsp.top_peaks(lags, r, n=1, min_x=lo, max_x=min(hi, len(x) * hop_s / 2), prominence=0.05)
    return (pk[0][0], pk[0][1]) if pk else (np.nan, np.nan)


def wobble(ctx, ch, reg):
    t0, t1, midi = reg
    sr = ctx.sr
    y = ctx.sig("gtr", ch)[int((t0 + 0.06) * sr):int(t1 * sr)]
    f_note = float(librosa.midi_to_hz(midi)) * 2 ** (ctx.tuning() / 12)
    hop = 128
    out = dict(start=t0, start_str=fmt_time(t0), length=t1 - t0, note=librosa.midi_to_note(int(midi)))
    try:
        f0 = librosa.yin(y, fmin=f_note * 2 ** (-1.5 / 12), fmax=f_note * 2 ** (1.5 / 12), sr=sr,
                         frame_length=2048, hop_length=hop)
    except Exception:
        f0 = np.array([])
    if len(f0) > 20:
        cents = 1200 * np.log2(f0 / np.median(f0))
        cents[np.abs(cents) > 150] = np.nan
        tt = np.arange(len(cents))
        ok = np.isfinite(cents)
        if ok.sum() > 20:
            p = np.polyfit(tt[ok], cents[ok], 1)
            cents = cents - np.polyval(p, tt)
            per, strength = _periodicity(cents, hop / sr, 0.08, 1.5)
            out.update(vib_period=per, vib_rate_hz=1 / per if per == per else np.nan, vib_strength=strength,
                       vib_depth_cents=float((np.nanpercentile(cents, 95) - np.nanpercentile(cents, 5)) / 2))
    env = ctx.env("gtr", ch)[ctx.env_index(t0 + 0.06):ctx.env_index(t1)]
    if len(env) > 20:
        tt = np.arange(len(env))
        e = env - np.polyval(np.polyfit(tt, env, 2), tt)
        per, strength = _periodicity(e, ctx.env_hop_s, 0.05, 1.5)
        out.update(trem_period=per, trem_rate_hz=1 / per if per == per else np.nan, trem_strength=strength,
                   trem_depth_db=float((np.percentile(e, 95) - np.percentile(e, 5)) / 2))
    return out


# ---- 図 ------------------------------------------------------------------
def plot_images(ctx, specs, t0, t1, title, path, valleys=None):
    fig, axes = plotting.plt.subplots(2, 2, figsize=(16, 9), squeeze=False,
                                      gridspec_kw=dict(hspace=0.35, wspace=0.12))
    for col, ch in enumerate(("L", "R")):
        sp = specs[ch]
        a, b, act, Zl, Zn = sp.section(t0, t1)
        ext_t = [sp.time_of(a), sp.time_of(b)]
        ax = axes[0, col]
        lim = plotting.robust_clim(Zl)
        ax.imshow(Zl, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                  extent=[ext_t[0], ext_t[1], 0, Zl.shape[0]], interpolation="nearest")
        ticks = [250, 500, 1000, 2000, 4000, 8000]
        ax.set_yticks([LOG_BPO * np.log2(f / LOG_FMIN) for f in ticks])
        ax.set_yticklabels(["250", "500", "1k", "2k", "4k", "8k"])
        if valleys and ch in valleys:
            f = valleys[ch]
            tt = sp.time_of(a + np.arange(len(f)))
            ax.plot(tt, LOG_BPO * np.log2(f / LOG_FMIN), ".", color="k", ms=1.2, alpha=0.6)
        ax.set_title(f"ギター{CH_JA[ch]}：対数周波数（谷の位置＝黒点）", fontsize=9, loc="left", pad=14)
        plotting.draw_grid(ax, ctx, t0, t1)
        ax = axes[1, col]
        lim = plotting.robust_clim(Zn)
        ax.imshow(Zn, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                  extent=[ext_t[0], ext_t[1], sp.lin_freqs[0], sp.lin_freqs[-1]], interpolation="nearest")
        ax.set_ylabel("Hz")
        ax.set_title(f"ギター{CH_JA[ch]}：線形周波数", fontsize=9, loc="left", pad=14)
        plotting.draw_grid(ax, ctx, t0, t1)
    fig.suptitle(title + "（赤＝いつもより強い、青＝谷）", fontsize=10)
    plotting.save(fig, path)


def _fmt_periods(pk, beat_s, bpb):
    return " / ".join(f"{p:.2f}秒（{p / beat_s:.1f}拍＝{p / beat_s / bpb:.2f}小節, {v:.2f}）" for p, v in pk) if pk else ""


def analyze_range(ctx, specs, t0, t1):
    res = {}
    for ch in ("L", "R"):
        sp = specs[ch]
        a, b, act, Zl, Zn = sp.section(t0, t1)
        d = dict(active_ratio=float(act.mean()) if len(act) else 0.0)
        if d["active_ratio"] < 0.1:
            res[ch] = d
            continue
        f, depth = valley_track(Zl, sp.centers, act)
        d["valley_f"] = f
        d["valley_p10"] = float(np.nanpercentile(f, 10))
        d["valley_p90"] = float(np.nanpercentile(f, 90))
        d["valley_median"] = float(np.nanmedian(f))
        d["valley_depth_db"] = float(np.nanmedian(depth))
        d["valley_periods"] = _period_peaks(np.log2(f), sp.hop_s, t1 - t0)
        d["row_periods"] = row_acf_peaks(Zl, sp.hop_s)
        d["chord_periods"] = chord_period(ctx, ch, t0, t1)
        d["notch_count"], d["notch_spacing_cv"] = notch_stats(Zn, sp.lin_freqs, act)
        warn = []
        for p, _ in (d["valley_periods"][:2] + d["row_periods"][:1]):
            for cp, _ in d["chord_periods"][:2]:
                for mult in (1, 2, 0.5):
                    if abs(p - cp * mult) / (cp * mult) < 0.08:
                        warn.append(f"揺れの候補{p:.2f}秒がコードの周期{cp:.2f}秒" + ("と同じ" if mult == 1 else f"の{mult}倍"))
        d["warning"] = "；".join(sorted(set(warn)))
        regs = sustained_regions(ctx, ch, t0, t1)
        d["regions"] = [wobble(ctx, ch, r) for r in regs]
        res[ch] = d
    return res


def run(ctx):
    out = ctx.subdir("modulation")
    secs = ctx.sections()
    print("[modulation] スペクトログラムを計算中")
    specs = {ch: ModSpec(ctx, "gtr", ch) for ch in ("L", "R")}
    beat_s, bpb = ctx.beat_s, ctx.beats_per_bar
    rows, reg_rows = [], []
    for s in secs:
        res = analyze_range(ctx, specs, s["start"], s["end"])
        for ch, d in res.items():
            row = dict(section=s["idx"], name=s["name"], start_str=fmt_time(s["start"]), channel=ch,
                       channel_ja=CH_JA[ch], active_ratio=d["active_ratio"])
            if "valley_f" in d:
                row.update(valley_p10_hz=d["valley_p10"], valley_median_hz=d["valley_median"],
                           valley_p90_hz=d["valley_p90"], valley_depth_db=d["valley_depth_db"],
                           valley_periods=_fmt_periods(d["valley_periods"], beat_s, bpb),
                           image_periods=_fmt_periods(d["row_periods"], beat_s, bpb),
                           chord_periods=_fmt_periods(d["chord_periods"], beat_s, bpb),
                           warning=d["warning"], notch_count=d["notch_count"],
                           notch_spacing_cv=d["notch_spacing_cv"])
                regs = d["regions"]
                for r in regs:
                    reg_rows.append(dict(section=s["idx"], channel=ch, **r))
                row["n_sustained"] = len(regs)
                for k in ("vib_rate_hz", "vib_depth_cents", "vib_strength", "trem_rate_hz", "trem_depth_db",
                          "trem_strength"):
                    row[k] = nan_median([r.get(k, np.nan) for r in regs])
                row["vib_periodic_ratio"] = float(np.mean([(r.get("vib_strength") or 0) >= 0.3 for r in regs])) if regs else np.nan
            rows.append(row)
        valleys = {ch: res[ch]["valley_f"] for ch in res if "valley_f" in res[ch]}
        if any(res[ch]["active_ratio"] >= 0.1 for ch in res):
            plot_images(ctx, specs, s["start"], s["end"], f"揺れ系の判定材料：{s['title']}",
                        out / f"{s['label']}.png", valleys)
    df = pd.DataFrame(rows)
    save_csv(df, out / "summary.csv")
    save_csv(pd.DataFrame(reg_rows), out / "sustained_notes.csv")
    save_json(out / "modulation.json", dict(reference=REFERENCE, sections=df.to_dict(orient="records")))
    print(f"[modulation] {len(secs)} 区間、伸ばした単音 {len(reg_rows)} 個")
    return df
