"""--check：指定した時間帯だけを詳しく調べる

出力（results/check/<範囲>/）
- pitch_LR.png：左右別の音高（CQT）の画像
- modulation_LR.png：周波数の山と谷の動き（対数・線形、左右別）
- rests.png：休符の残り方（左右の音量カーブと、休符ごとの減衰）
- delay.png：この範囲だけで出したディレイの曲線
- check.md / rests.csv：数値
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import delay, modulation, plotting
from .common import CH_JA, CQT_BINS, CQT_FMIN_MIDI, db, fmt_time, save_csv


def _range_dir(ctx, t0, t1):
    def f(x):
        m = int(x // 60)
        return f"{m}m{x - 60 * m:04.1f}s"
    return ctx.subdir("check") / f"{f(t0)}-{f(t1)}"


def run(ctx, t0, t1):
    t1 = min(t1, ctx.duration)
    out = _range_dir(ctx, t0, t1)
    out.mkdir(parents=True, exist_ok=True)
    title = f"{fmt_time(t0, 1)}〜{fmt_time(t1, 1)}"
    print(f"[check] {title} を調べています → {out}")

    # ---- 左右別の音高 ----
    fig, axes = plotting.plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw=dict(hspace=0.25))
    a, b = ctx.frame_index(t0), ctx.frame_index(t1)
    for ax, ch in zip(axes, ("L", "R")):
        C = db(ctx.cqt("gtr", ch)[:, a:b])
        vmax = np.percentile(C, 99.5)
        ax.imshow(C, origin="lower", aspect="auto", cmap="magma", vmin=vmax - 60, vmax=vmax,
                  extent=[a * ctx.hop_s, b * ctx.hop_s, -0.5, CQT_BINS - 0.5], interpolation="nearest")
        plotting.note_axis(ax, CQT_FMIN_MIDI, CQT_FMIN_MIDI + CQT_BINS - 1, CQT_FMIN_MIDI)
        ax.set_ylim(-0.5, 60)
        ax.set_title(f"ギター{CH_JA[ch]}：音高（半音ごと、明るい＝強い）", fontsize=9, loc="left", pad=14)
        plotting.draw_grid(ax, ctx, t0, t1)
    plotting.save(fig, out / "pitch_LR.png")

    # ---- 山と谷の動き ----
    pad = 1.0
    specs = {ch: modulation.ModSpec(ctx, "gtr", ch, t0 - pad, t1 + pad) for ch in ("L", "R")}
    mres = modulation.analyze_range(ctx, specs, t0, t1)
    modulation.plot_images(ctx, specs, t0, t1, f"揺れ系の判定材料：{title}", out / "modulation_LR.png",
                           {ch: mres[ch]["valley_f"] for ch in mres if "valley_f" in mres[ch]})

    # ---- 休符の残り方とディレイ ----
    dres = delay.analyze_range(ctx, t0, t1)
    delay.plot_range(ctx, dres, f"ディレイの判定材料：{title}", out / "delay.png")
    fig, axes = plotting.plt.subplots(2, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=[3, 1], hspace=0.35))
    hop = ctx.env_hop_s
    rows = []
    for r_, ch in enumerate(("L", "R")):
        env = ctx.env("gtr", ch)
        ia, ib = ctx.env_index(t0), ctx.env_index(t1)
        tt = np.arange(ia, ib) * hop
        ax = axes[r_, 0]
        ax.plot(tt, env[ia:ib], color="tab:blue" if ch == "L" else "tab:red", lw=0.7)
        ax2 = axes[r_, 1]
        for k, r in enumerate(dres[ch]["rests"]):
            ax.axvspan(r["stop"], r["end"], color="gold", alpha=0.3)
            ax.plot([r["stop"] + 0.08], [r["ref_db"] - (r["drop80"] if np.isfinite(r["drop80"]) else np.nan)],
                    "kv", ms=4)
            i0, i1 = ctx.env_index(r["stop"]), ctx.env_index(r["end"])
            ax2.plot((np.arange(i0, i1) - i0) * hop * 1000, env[i0:i1] - r["ref_db"], lw=0.8)
            rows.append(dict(channel=ch, stop=fmt_time(r["stop"]), rest_len_ms=r["rest_len"] * 1000,
                             ref_db=r["ref_db"], drop50=r["drop50"], drop80=r["drop80"]))
        ax.set_ylim(max(np.min(env[ia:ib]), -100), np.max(env[ia:ib]) + 3)
        ax.set_ylabel("dBFS")
        ax.set_title(f"ギター{CH_JA[ch]}：音量（黄＝休符、▼＝止めて80ms後）", fontsize=9, loc="left", pad=14)
        plotting.draw_grid(ax, ctx, t0, t1)
        ax2.axvline(80, color="k", ls=":", lw=0.7)
        ax2.set_xlabel("止めてからの時間（ms）")
        ax2.set_ylabel("直前の音＝0dB")
        ax2.set_ylim(-70, 3)
        ax2.grid(alpha=0.3)
        ax2.set_title(f"休符ごとの減衰（{len(dres[ch]['rests'])}個）", fontsize=9)
    plotting.save(fig, out / "rests.png")
    save_csv(pd.DataFrame(rows), out / "rests.csv")

    # ---- 数値のまとめ ----
    beat_s, bpb = ctx.beat_s, ctx.beats_per_bar
    lines = [f"# 詳細チェック {title}", "",
             f"テンポ {ctx.tempo:.1f} BPM（4分 {beat_s * 1000:.0f}ms、付点8分 {0.75 * beat_s * 1000:.0f}ms、"
             f"8分 {0.5 * beat_s * 1000:.0f}ms）", "",
             "## 休符の残り方とディレイ", "",
             "| チャンネル | 休符 | 80ms後の落ち幅 中央値 | 休符：過去−未来の山 | 切り替わり：過去−未来の山 | 反対チャンネルの山 | モデル最大 fb0 / 0.3 / 0.5 |",
             "|---|---|---|---|---|---|---|"]
    for ch in ("L", "R", "sum"):
        d = dres[ch]
        mb = d["model_best"]
        lines.append(f"| {CH_JA[ch]} | {d['n_rests']} | {d['drop80_median']:.1f} dB (n={d['drop80_n']}) | "
                     f"{delay._fmt_peaks(d.get('rest_diff_peaks'))} | {delay._fmt_peaks(d.get('note_diff_peaks'))} | "
                     f"{delay._fmt_peaks(d.get('cross_diff_peaks'))} | "
                     + " / ".join(f"{mb[fb][0] * 1000:.0f}ms" if fb in mb else "-" for fb in delay.FBS) + " |")
    lines += ["", "## 山と谷の動き", "",
              "| チャンネル | 谷の範囲（10〜90%） | 谷の周期の候補 | 画像の周期の候補 | コードの周期 | 注意 | 線形での谷の数 | 単音の音高の揺れ | 音量の揺れ |",
              "|---|---|---|---|---|---|---|---|---|"]
    for ch in ("L", "R"):
        d = mres[ch]
        if "valley_f" not in d:
            lines.append(f"| {CH_JA[ch]} | ギターがほぼ鳴っていない |||||||| ")
            continue
        regs = d["regions"]
        vib = " / ".join(f"{r['start_str']} {r.get('vib_rate_hz', np.nan):.1f}Hz ±{r.get('vib_depth_cents', np.nan):.0f}c"
                         for r in regs[:3])
        trem = " / ".join(f"{r.get('trem_rate_hz', np.nan):.1f}Hz ±{r.get('trem_depth_db', np.nan):.1f}dB" for r in regs[:3])
        lines.append(f"| {CH_JA[ch]} | {d['valley_p10']:.0f}〜{d['valley_p90']:.0f} Hz | "
                     f"{modulation._fmt_periods(d['valley_periods'], beat_s, bpb)} | "
                     f"{modulation._fmt_periods(d['row_periods'], beat_s, bpb)} | "
                     f"{modulation._fmt_periods(d['chord_periods'], beat_s, bpb)} | {d['warning']} | "
                     f"{d['notch_count']:.0f} | {vib} | {trem} |")
    lines += ["", "画像：pitch_LR.png（音高）、modulation_LR.png（山と谷）、rests.png（休符）、delay.png（ディレイの曲線）",
              "", "※ 数値は判断材料です。ここでは「あり／なし」を決めていません。"]
    (out / "check.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[check] 完了：{out / 'check.md'}")
    return out
