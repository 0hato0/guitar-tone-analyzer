"""stomp（踏める場所）

- 各パートの切れ目の前後2秒で、ギターの右・左・合計それぞれ一番静かな0.15秒の区間とその音量
- ギターが完全に無音（左右とも -60dBFS 以下）の区間を曲全体から探し、優先して表示する
  （原曲も無音か＝全員が止まっているか、も付ける）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import dsp, plotting
from .common import CH_JA, fmt_time, save_csv, save_json

WIN_S = 0.15
AROUND_S = 2.0
SILENT_DB = -60.0


def _levels(ctx, kind, ch):
    return dsp.moving_level_db(ctx.sig(kind, ch), ctx.sr, WIN_S)


def silent_intervals(t, lv, thr=SILENT_DB):
    """「窓の平均が thr 以下」のフレームが続く区間（窓の長さを含めた実際の時間）"""
    quiet = lv <= thr
    out = []
    i = 0
    n = len(quiet)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            out.append((float(t[i]), float(t[j - 1] + WIN_S)))
            i = j
        else:
            i += 1
    return out


def run(ctx):
    out = ctx.subdir("stomp")
    secs = ctx.sections()
    lv = {ch: _levels(ctx, "gtr", ch) for ch in ("L", "R", "sum")}
    t = lv["sum"][0]
    gmax = np.maximum(lv["L"][1], lv["R"][1])
    _, mix_lv = _levels(ctx, "mix", "sum")
    g_first = ctx.beat_info()["guitar_first_onset"]

    # 完全に無音の区間
    sil = []
    for a, b in silent_intervals(t, gmax):
        ia, ib = np.searchsorted(t, a), max(np.searchsorted(t, b - WIN_S), np.searchsorted(t, a) + 1)
        mix_min = float(np.min(mix_lv[ia:ib])) if ib > ia else np.nan
        near = min(secs, key=lambda s: abs(s["start"] - a))
        sil.append(dict(start=a, end=b, start_str=fmt_time(a), end_str=fmt_time(b), length=b - a,
                        mix_min_db=mix_min, mix_silent=bool(mix_min <= SILENT_DB),
                        before_guitar=bool(b <= g_first + 0.05),
                        nearest_section=near["idx"], nearest_boundary=fmt_time(near["start"]),
                        from_boundary=a - near["start"]))
    sil_df = pd.DataFrame(sil)
    save_csv(sil_df, out / "silence.csv")

    # 切れ目の前後で一番静かな0.15秒
    rows = []
    for s in secs[1:]:
        b = s["start"]
        sel = (t >= b - AROUND_S) & (t + WIN_S <= b + AROUND_S)
        if not sel.any():
            continue
        row = dict(section=s["idx"], name=s["name"], boundary=b, boundary_str=fmt_time(b))
        for ch in ("R", "L", "sum"):
            v = lv[ch][1][sel]
            i = int(np.argmin(v))
            ts = float(t[sel][i])
            row[f"{ch}_quiet_start"] = fmt_time(ts)
            row[f"{ch}_quiet_end"] = fmt_time(ts + WIN_S)
            row[f"{ch}_quiet_db"] = float(v[i])
            row[f"{ch}_from_boundary"] = ts - b
        both = gmax[sel]
        i = int(np.argmin(both))
        ts = float(t[sel][i])
        row["LR_quiet_start"] = fmt_time(ts)
        row["LR_quiet_db"] = float(both[i])
        near_sil = [x for x in sil if x["end"] >= b - AROUND_S and x["start"] <= b + AROUND_S]
        row["silent_nearby"] = " ".join(f"{x['start_str']}〜{x['end_str']}" for x in near_sil)
        rows.append(row)
    df = pd.DataFrame(rows)
    save_csv(df, out / "stomp.csv")
    save_json(out / "stomp.json", dict(silent_db=SILENT_DB, window_s=WIN_S,
                                       silence=sil, boundaries=df.to_dict(orient="records")))

    # 図：曲全体のギター音量と無音区間
    fig, ax = plotting.plt.subplots(figsize=(16, 4))
    ax.plot(t, lv["L"][1], color="tab:blue", lw=0.5, label="ギター左")
    ax.plot(t, lv["R"][1], color="tab:red", lw=0.5, label="ギター右")
    ax.plot(t, mix_lv, color="0.6", lw=0.4, label="原曲")
    for x in sil:
        ax.axvspan(x["start"], x["end"], color="tab:green" if not x["mix_silent"] else "gold", alpha=0.35)
    ax.axhline(SILENT_DB, color="k", lw=0.6, ls=":")
    ax.set_ylim(-100, 0)
    ax.set_ylabel("0.15秒平均 dBFS")
    ax.set_title("踏める場所：緑＝ギターが -60dB 以下、黄＝原曲も -60dB 以下")
    ax.legend(fontsize=7, loc="lower right")
    plotting.draw_grid(ax, ctx, 0, ctx.duration, labels=True)
    plotting.save(fig, out / "stomp.png")
    print(f"[stomp] 無音区間 {len(sil)} 個 / 切れ目 {len(rows)} 個")
    return df
