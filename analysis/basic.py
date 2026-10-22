"""basic（基本情報）：長さ・サンプルレート・チャンネル数、テンポ（全体と20秒ごと）、チューニング、最低音。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import librosa

from . import plotting
from .common import (CQT_FMIN_MIDI, HOP, fmt_time, midi_name, save_csv, save_json)

WINDOW_S = 20.0


def run(ctx):
    out = ctx.subdir("basic")
    info = ctx.audio.info
    bi = ctx.beat_info()
    tempo = float(bi["tempo"])
    beats = bi["beats"][~bi["extrapolated"]]

    # ---- 20秒ごとのテンポ ----
    env = ctx.onset_env("mix", "sum")
    rows = []
    t = 0.0
    while t < ctx.duration - 2:
        a, b = int(t / ctx.hop_s), int(min(t + WINDOW_S, ctx.duration) / ctx.hop_s)
        seg = env[a:b]
        w_ac = float(librosa.feature.tempo(onset_envelope=seg, sr=ctx.sr, hop_length=HOP,
                                           start_bpm=tempo)[0]) if len(seg) > 50 else np.nan
        bsel = beats[(beats >= t) & (beats < t + WINDOW_S)]
        w_beat = 60.0 / float(np.median(np.diff(bsel))) if len(bsel) >= 4 else np.nan
        note = ""
        for v, src in ((w_ac, "自己相関"), (w_beat, "拍の間隔")):
            if not np.isfinite(v):
                continue
            r = v / tempo
            if abs(r - 2) < 0.12 or abs(r - 0.5) < 0.06 or abs(r - 1.5) < 0.08 or abs(r - 2 / 3) < 0.05:
                note += f"{src}が全体の{r:.2f}倍（拍の取り違えの可能性） "
            elif abs(r - 1) > 0.03:
                note += f"{src}が全体から{(r - 1) * 100:+.1f}% "
        rows.append(dict(start=t, end=min(t + WINDOW_S, ctx.duration), start_str=fmt_time(t, 0),
                         tempo_autocorr=w_ac, tempo_beats=w_beat, n_beats=len(bsel), note=note.strip()))
        t += WINDOW_S
    tw = pd.DataFrame(rows)
    save_csv(tw, out / "tempo_windows.csv")

    # ---- チューニング ----
    tuning = ctx.tuning()
    cents = tuning * 100.0
    a4 = 440.0 * 2 ** (cents / 1200)
    try:
        mix_cents = float(librosa.estimate_tuning(y=ctx.sig("mix", "sum"), sr=ctx.sr, n_fft=4096)) * 100
    except Exception:
        mix_cents = np.nan

    # ---- 最低音：低音域で一番低い音の分布 ----
    low_rows = []
    lowest_est = {}
    for ch in ("sum", "L", "R"):
        _, low = ctx.notes("gtr", ch)
        v = low[np.isfinite(low)]
        if len(v) == 0:
            lowest_est[ch] = None
            continue
        vals, counts = np.unique(v.astype(int), return_counts=True)
        share = counts / counts.sum()
        for m, c, s in zip(vals, counts, share):
            low_rows.append(dict(channel=ch, midi=int(m), note=midi_name(m), frames=int(c), share=float(s)))
        # 鳴っているフレームの0.5%以上で「一番低い音」になった音のうち最も低いもの
        ok = vals[share >= 0.005]
        lowest_est[ch] = midi_name(ok.min()) if len(ok) else midi_name(vals.min())
    low_df = pd.DataFrame(low_rows)
    save_csv(low_df, out / "low_notes.csv")

    res = {
        "files": info["files"],
        "original": info["original"],
        "analysis_sr": info["analysis_sr"],
        "duration_s": info["duration_s"],
        "duration": fmt_time(info["duration_s"], 1),
        "offset": info["offset"],
        "guitar_gain_ls": info["guitar_gain_ls"],
        "guitar_gain_used": info["guitar_gain_used"],
        "tempo_bpm": tempo,
        "tempo_windows_range": [float(np.nanmin(tw["tempo_beats"])) if len(tw) else None,
                                float(np.nanmax(tw["tempo_beats"])) if len(tw) else None],
        "tuning_cents": cents,
        "tuning_a4_hz": a4,
        "tuning_cents_mix": mix_cents,
        "lowest_note": lowest_est,
        "beat_track": {"n_beats": int(len(beats)),
                       "first_beat": float(beats[0]) if len(beats) else None,
                       "last_beat": float(beats[-1]) if len(beats) else None},
    }
    save_json(out / "basic.json", res)

    # ---- 図 ----
    fig, axes = plotting.plt.subplots(2, 1, figsize=(12, 6.5), gridspec_kw=dict(hspace=0.45))
    ax = axes[0]
    if len(beats) > 2:
        ibi = np.diff(beats)
        loc = 60.0 / ibi
        ax.plot(beats[1:], loc, ".", ms=2, color="0.6", label="拍ごと")
        k = 9
        if len(loc) > k:
            sm = np.convolve(loc, np.ones(k) / k, mode="same")
            ax.plot(beats[1:], sm, "-", color="tab:blue", lw=1, label="拍ごと（9拍平均）")
    ax.step(tw["start"], tw["tempo_autocorr"], where="post", color="tab:orange", label="20秒ごと（自己相関）")
    ax.axhline(tempo, color="k", ls="--", lw=0.8, label=f"全体 {tempo:.1f}")
    ax.set_ylim(tempo * 0.4, tempo * 2.2)
    ax.set_ylabel("BPM")
    ax.set_title(f"テンポ（全体 {tempo:.1f} BPM）")
    ax.legend(fontsize=7, loc="upper right")
    plotting.draw_grid(ax, ctx, 0, ctx.duration, labels=False)

    ax = axes[1]
    if len(low_df):
        for ch, c in (("sum", "k"), ("L", "tab:blue"), ("R", "tab:red")):
            d = low_df[low_df.channel == ch]
            ax.plot(d["midi"], d["share"] * 100, "-o", ms=3, color=c, label={"sum": "合計", "L": "左", "R": "右"}[ch])
        lo = int(low_df["midi"].min())
        hi = int(min(low_df["midi"].max(), lo + 36))
        ax.set_xticks(range(lo, hi + 1, 1))
        ax.set_xticklabels([midi_name(m) for m in range(lo, hi + 1)], rotation=90, fontsize=7)
        ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylabel("一番低い音になった割合 (%)")
    ax.set_title(f"低音域の音の分布（推定の最低音：合計 {lowest_est.get('sum')}）  チューニング {cents:+.1f}セント（A={a4:.1f}Hz）")
    ax.legend(fontsize=7)
    plotting.save(fig, out / "basic.png")

    print(f"[basic] テンポ {tempo:.1f} BPM / チューニング {cents:+.1f} セント / 最低音 {lowest_est.get('sum')}")
    return res
