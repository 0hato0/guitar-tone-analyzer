"""features（パートごとの数値）：パートごと・合計／左／右ごとの音量、GEQ7バランス、左右、音高、最低音、
クレストファクター、16分のグリッドごとの立ち上がりの強さ。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import librosa

from . import dsp, plotting
from .common import (CH_JA, CHANNELS, CQT_FMIN_MIDI, GEQ7_CENTERS, GEQ7_LABELS,
                     NOTE_NAMES, corr, db, fmt_time, midi_name, nan_median,
                     pow_db, rms_db, save_csv, save_json)

GRID_LABELS = ["表", "16分裏(1つ目)", "8分裏", "16分裏(2つ目)"]
GRID_KEYS = ["g1", "g_e", "g_and", "g_a"]
N_FFT = 4096
STFT_HOP = 2048


def crest_db(y, sr, active_thr_db):
    """50msごとの（ピーク÷RMS）の中央値（鳴っているブロックだけ）"""
    n = int(0.05 * sr)
    k = len(y) // n
    if k == 0:
        return np.nan
    B = y[:k * n].reshape(k, n).astype(np.float64)
    rms = np.sqrt(np.mean(B ** 2, axis=1))
    pk = np.max(np.abs(B), axis=1)
    sel = db(rms) > active_thr_db
    if sel.sum() < 3:
        return np.nan
    return float(np.median(db(pk[sel]) - db(rms[sel])))


def grid_strength(onset_env, hop_s, beats, t0, t1):
    """拍を4等分した位置ごとの立ち上がりの強さ（平均）。拍の位置はビート追跡の結果"""
    vals = [[] for _ in range(4)]
    w = max(1, int(round(0.025 / hop_s)))
    for b0, b1 in zip(beats[:-1], beats[1:]):
        if b0 < t0 or b1 > t1:
            continue
        for k in range(4):
            p = b0 + (b1 - b0) * k / 4
            i = int(round(p / hop_s))
            seg = onset_env[max(0, i - w): i + w + 1]
            if len(seg):
                vals[k].append(float(seg.max()))
    m = np.array([np.mean(v) if v else np.nan for v in vals])
    return m


def run(ctx):
    out = ctx.subdir("features")
    secs = ctx.sections()
    sr = ctx.sr
    beats = ctx.beats()
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    oth_sum = ctx.sig("oth", "sum")
    thr_g = ctx.active_threshold("gtr")

    rows = []
    for ch in CHANNELS:
        y = ctx.sig("gtr", ch)
        P = dsp.stft_power(y, N_FFT, STFT_HOP)
        C = ctx.cqt("gtr", ch)
        chroma = ctx.chroma("gtr", ch)
        act = ctx.active_frames("gtr", ch)
        count, lowest = ctx.notes("gtr", ch)
        oenv = ctx.onset_env("gtr", ch)
        env = ctx.env("gtr", ch)
        for s in secs:
            a, e = int(s["start"] * sr), int(s["end"] * sr)
            fa, fe = ctx.frame_index(s["start"]), ctx.frame_index(s["end"])
            ea, ee = ctx.env_index(s["start"]), ctx.env_index(s["end"])
            act_ratio = float(act[fa:fe].mean()) if fe > fa else 0.0
            lvl = rms_db(y[a:e])
            env_s = env[ea:ee]
            on = env_s > thr_g
            lvl_act = float(pow_db(np.mean(10 ** (env_s[on] / 10)))) if on.sum() > 3 else np.nan
            oth_lvl = rms_db(oth_sum[a:e])
            row = dict(section=s["idx"], name=s["name"], start=s["start"], end=s["end"],
                       start_str=fmt_time(s["start"]), channel=ch, channel_ja=CH_JA[ch],
                       active_ratio=act_ratio, level_db=lvl, level_active_db=lvl_act,
                       other_db=oth_lvl, vs_other_db=lvl - oth_lvl)
            # 左右の音量差と相関（どのチャンネルの行にも同じ値）
            L, R = ctx.sig("gtr", "L")[a:e], ctx.sig("gtr", "R")[a:e]
            row["R_minus_L_db"] = rms_db(R) - rms_db(L)
            row["LR_corr"] = corr(L, R)
            guitar_here = act_ratio >= 0.05
            row["guitar_present"] = guitar_here
            # GEQ7
            pa, pe = int(s["start"] * sr / STFT_HOP), int(s["end"] * sr / STFT_HOP)
            if guitar_here and pe > pa:
                bp = dsp.band_powers(P[:, pa:pe].mean(axis=1), freqs, GEQ7_CENTERS)
                rel = bp - bp.max()
                for lab, v in zip(GEQ7_LABELS, rel):
                    row[f"geq_{lab}"] = float(v)
                row["geq_peak"] = GEQ7_LABELS[int(np.argmax(bp))]
            else:
                for lab in GEQ7_LABELS:
                    row[f"geq_{lab}"] = np.nan
                row["geq_peak"] = ""
            # 音高
            sel = act[fa:fe]
            if guitar_here and sel.sum() > 5:
                cv = chroma[:, fa:fe][:, sel].mean(axis=1)
                order = np.argsort(-cv)
                row["main_pitch"] = " ".join(NOTE_NAMES[i] for i in order[:3])
                Cs = C[:, fa:fe][:, sel]
                w = Cs.sum(axis=1)
                midi = CQT_FMIN_MIDI + np.arange(len(w))
                cum = np.cumsum(w) / w.sum()
                row["register_low"] = midi_name(midi[np.searchsorted(cum, 0.1)])
                row["register_mid"] = midi_name(midi[np.searchsorted(cum, 0.5)])
                row["register_high"] = midi_name(midi[np.searchsorted(cum, 0.9)])
                cnt = count[fa:fe][sel]
                cnt = cnt[np.isfinite(cnt)]
                med = float(np.median(cnt)) if len(cnt) else np.nan
                ratio3 = float(np.mean(cnt >= 3)) if len(cnt) else np.nan
                row["notes_median"] = med
                row["chord_frames_ratio"] = ratio3
                if not np.isfinite(med):
                    row["mono_or_chord"] = ""
                elif ratio3 >= 0.4 or med >= 3:
                    row["mono_or_chord"] = "コード寄り"
                elif med <= 1 and ratio3 < 0.15:
                    row["mono_or_chord"] = "単音寄り"
                else:
                    row["mono_or_chord"] = "中間（パワーコード・重音など）"
                lw = lowest[fa:fe][sel]
                lw = lw[np.isfinite(lw)]
                row["lowest_note"] = midi_name(np.percentile(lw, 3)) if len(lw) else ""
            else:
                for k in ("main_pitch", "register_low", "register_mid", "register_high", "mono_or_chord", "lowest_note"):
                    row[k] = ""
                row["notes_median"] = np.nan
                row["chord_frames_ratio"] = np.nan
            row["crest_db"] = crest_db(y[a:e], sr, thr_g) if guitar_here else np.nan
            if guitar_here:
                g = grid_strength(oenv, ctx.hop_s, beats, s["start"], s["end"])
                gm = np.nanmax(g) if np.any(np.isfinite(g)) else np.nan
                for key, v in zip(GRID_KEYS, g):
                    row[key] = float(v / gm) if gm and np.isfinite(gm) else np.nan
            else:
                for key in GRID_KEYS:
                    row[key] = np.nan
            rows.append(row)
        del P
    df = pd.DataFrame(rows)
    save_csv(df, out / "features.csv")
    save_json(out / "features.json", df.to_dict(orient="records"))

    # ---- 図：GEQ7バランス ----
    fig, axes = plotting.plt.subplots(1, 3, figsize=(15, 0.45 * len(secs) + 2), sharey=True)
    for ax, ch in zip(axes, CHANNELS):
        d = df[df.channel == ch]
        M = d[[f"geq_{l}" for l in GEQ7_LABELS]].to_numpy(float)
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=-24, vmax=0)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="w" if M[i, j] < -12 else "k")
        ax.set_xticks(range(7))
        ax.set_xticklabels(GEQ7_LABELS)
        ax.set_title(f"ギター{CH_JA[ch]}：GEQ7の帯域バランス（一番強い帯域＝0dB）", fontsize=9)
        ax.set_yticks(range(len(secs)))
        ax.set_yticklabels([s["title"] for s in secs], fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.6, label="dB")
    plotting.save(fig, out / "geq7.png")

    # ---- 図：音量と16分のグリッド ----
    fig, axes = plotting.plt.subplots(1, 2, figsize=(15, 0.45 * len(secs) + 2), sharey=True)
    ax = axes[0]
    y_ = np.arange(len(secs))
    for ch, c, off in (("sum", "k", 0), ("L", "tab:blue", -0.2), ("R", "tab:red", 0.2)):
        d = df[df.channel == ch]
        ax.barh(y_ + off, np.maximum(d.level_db, -80) + 80, left=-80, height=0.2, color=c, label=f"ギター{CH_JA[ch]}")
    d = df[df.channel == "sum"]
    ax.plot(np.maximum(d.other_db, -80), y_, "D", color="tab:green", ms=4, label="他の楽器")
    ax.set_yticks(y_)
    ax.set_yticklabels([s["title"] for s in secs], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("音量 dBFS（区間の平均）")
    ax.legend(fontsize=7)
    ax = axes[1]
    for ch, c, off in (("L", "tab:blue", -0.15), ("R", "tab:red", 0.15)):
        d = df[df.channel == ch]
        G = d[GRID_KEYS].to_numpy(float)
        for k in range(4):
            ax.barh(y_ + off, np.nan_to_num(G[:, k]) * 0.22, left=k, height=0.28, color=c, alpha=0.8,
                    label=f"ギター{CH_JA[ch]}" if k == 0 else None)
    ax.set_xticks([0.1, 1.1, 2.1, 3.1])
    ax.set_xticklabels(GRID_LABELS, fontsize=8)
    ax.set_title("拍の中の位置ごとの立ち上がりの強さ（区間内で最大＝1）", fontsize=9)
    ax.legend(fontsize=7)
    plotting.save(fig, out / "levels_grid.png")

    print(f"[features] {len(secs)} 区間 × 合計/左/右")
    return df
