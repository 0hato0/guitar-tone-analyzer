"""reverb（リバーブの判定材料）

- ギターを止めた場所（0.15秒以内に20dB以上落ち、0.4秒以上戻らない場所）をすべて見つける
- それぞれ：止めてから20〜600msの減衰カーブ（直前の音＝0dB）、80・150・250・350ms後の値、
  100〜300msの傾き（dB/秒）、折れ曲がり（急に落ちてから緩やかに減る形か）
- 余韻（100〜350ms）の左右相関と、直前の音の左右相関
- 余韻の音高が直前の音と一致するか（クロマの一致度）
- 他の楽器がほぼ鳴っていない場面と、全員が止まる場面（キメ・曲の最後）は、原曲そのものでも同じ値を出し、
  左右の差の成分（L−R）の残り方も出す
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import dsp, plotting
from .common import CH_JA, CHANNELS, cos_sim, corr, fmt_time, nan_median, save_csv, save_json

OFFSETS_MS = np.arange(20, 601, 10)
POINTS_MS = (80, 150, 250, 350)
OTHER_QUIET_DB = 15.0   # 他の楽器がギターの直前の音よりこれ以上小さければ「ほぼ鳴っていない」

REFERENCE = {
    "v80": "止めて80ms後の値（直前の音＝0dB）。-50dBより下ならほぼ残っていない",
    "slope": "100〜300msの傾き。ホール・ルームの余韻は数十〜100dB/秒程度でゆっくり減ることが多い（目安）",
    "knee": "急に落ちてから緩やかに減る折れ曲がりがあれば、その後ろの部分が余韻の候補。ただしバンド中では他の楽器の響きと区別できない",
    "tail_corr": "余韻の左右相関が直前の音より下がっていれば、ステレオの残響の手がかり",
}


def _curve(env, hop, stop_i, ref, back_i):
    vals = np.full(len(OFFSETS_MS), np.nan)
    lim = back_i - int(round(0.02 / hop)) if back_i is not None else len(env)
    for k, ms in enumerate(OFFSETS_MS):
        i = stop_i + int(round(ms / 1000 / hop))
        if i < min(lim, len(env)):
            vals[k] = env[i] - ref
    return vals


def _points(curve):
    return {f"v{ms}": float(curve[np.argmin(np.abs(OFFSETS_MS - ms))]) for ms in POINTS_MS}


def _slope(curve):
    t = OFFSETS_MS / 1000
    sel = (t >= 0.1) & (t <= 0.3) & np.isfinite(curve)
    if sel.sum() < 5:
        return np.nan
    return float(np.polyfit(t[sel], curve[sel], 1)[0])


class _Signals:
    def __init__(self, ctx):
        self.ctx = ctx
        self.bp = {}

    def bandpassed(self, kind, ch):
        if (kind, ch) not in self.bp:
            self.bp[(kind, ch)] = dsp.bandpass(self.ctx.sig(kind, ch), self.ctx.sr, 100, 8000)
        return self.bp[(kind, ch)]

    def lr_corr(self, kind, t0, t1):
        sr = self.ctx.sr
        a, b = int(t0 * sr), int(t1 * sr)
        if a < 0 or b > len(self.ctx.sig(kind, "L")):
            return np.nan
        return corr(self.bandpassed(kind, "L")[a:b], self.bandpassed(kind, "R")[a:b])


def analyze_stops(ctx, sigs, ch, t0, t1):
    env = ctx.env("gtr", ch)
    hop = ctx.env_hop_s
    min_ref = ctx.active_threshold("gtr") + 10
    stops = dsp.find_stops(env, hop, ctx.env_index(t0), ctx.env_index(t1), drop_db=20, within=0.15,
                           back_db=10, quiet=0.4, min_ref=min_ref)
    chroma = ctx.chroma("gtr", ch)
    env_oth = ctx.env("oth", "sum")
    rows = []
    for st in stops:
        ts = st["stop"] * hop
        cv = _curve(env, hop, st["stop"], st["ref"], st["back"])
        row = dict(channel=ch, stop=ts, stop_str=fmt_time(ts), ref_db=st["ref"],
                   quiet_len=(st["back"] - st["stop"]) * hop, curve=cv)
        row.update(_points(cv))
        row["slope_100_300"] = _slope(cv)
        row.update(dsp.knee_fit(OFFSETS_MS / 1000, cv))
        row["tail_LR_corr"] = sigs.lr_corr("gtr", ts + 0.10, ts + 0.35)
        row["pre_LR_corr"] = sigs.lr_corr("gtr", ts - 0.20, ts - 0.02)
        fa, fb = ctx.frame_index(ts - 0.20), ctx.frame_index(ts - 0.02)
        ta, tb = ctx.frame_index(ts + 0.10), ctx.frame_index(ts + 0.35)
        if fa >= 0 and tb <= chroma.shape[1]:
            row["tail_chroma_match"] = cos_sim(chroma[:, ta:tb].mean(axis=1), chroma[:, fa:fb].mean(axis=1))
        else:
            row["tail_chroma_match"] = np.nan
        # 他の楽器はどれくらい鳴っているか
        oa, ob = ctx.env_index(ts - 0.3), ctx.env_index(ts + 0.6)
        row["other_rel_db"] = float(np.max(env_oth[max(oa, 0):ob]) - st["ref"]) if ob > oa else np.nan
        row["band_quiet"] = bool(row["other_rel_db"] < -OTHER_QUIET_DB)
        # 全員が止まったか（原曲も0.15秒で20dB以上落ちる）
        env_mix = ctx.env("mix", ch)
        ma = ctx.env_index(ts - 0.2)
        mix_ref = float(np.max(env_mix[max(ma, 0):st["stop"] + 1]))
        i15 = st["stop"] + int(round(0.15 / hop))
        row["all_stop"] = bool(i15 < len(env_mix) and env_mix[i15] <= mix_ref - 20)
        if row["band_quiet"] or row["all_stop"]:
            # 原曲が一度 -10dB より下がってから、また戻るところまでを使う
            lim = min(len(env_mix), st["stop"] + int(round(0.62 / hop)))
            below = np.flatnonzero(env_mix[st["stop"]:lim] < mix_ref - 10)
            back = lim
            if len(below):
                back = st["stop"] + int(below[0])
                while back < lim and env_mix[back] < mix_ref - 10:
                    back += 1
            mc = _curve(env_mix, hop, st["stop"], mix_ref, back)
            row["mix_curve"] = mc
            for k, v in _points(mc).items():
                row[f"mix_{k}"] = v
            row["mix_slope_100_300"] = _slope(mc)
            mk = dsp.knee_fit(OFFSETS_MS / 1000, mc)
            row["mix_knee_time"] = mk["knee_time"]
            row["mix_knee_level"] = mk["knee_level"]
            row["mix_slope_after"] = mk["slope_after"]
            row["mix_is_knee"] = mk["is_knee"]
            env_side = ctx.env("mix", "side")
            sc = _curve(env_side, hop, st["stop"], mix_ref, back)
            row["side_curve"] = sc
            for k, v in _points(sc).items():
                row[f"side_{k}"] = v
            sa = ctx.env_index(ts - 0.2)
            row["side_pre_db"] = float(np.max(env_side[max(sa, 0):st["stop"] + 1]) - mix_ref)
            row["mix_tail_LR_corr"] = sigs.lr_corr("mix", ts + 0.10, ts + 0.35)
            row["mix_pre_LR_corr"] = sigs.lr_corr("mix", ts - 0.20, ts - 0.02)
        rows.append(row)
    return rows


def run(ctx):
    out = ctx.subdir("reverb")
    secs = ctx.sections()
    sigs = _Signals(ctx)
    all_rows = []
    for ch in CHANNELS:
        for s in secs:
            for r in analyze_stops(ctx, sigs, ch, s["start"], s["end"]):
                r.update(section=s["idx"], name=s["name"])
                all_rows.append(r)
    if all_rows:
        last = max(r["stop"] for r in all_rows if r["channel"] == "sum") if any(
            r["channel"] == "sum" for r in all_rows) else None
        for r in all_rows:
            r["is_last"] = bool(last is not None and r["channel"] == "sum" and abs(r["stop"] - last) < 1e-6)

    flat_cols = [k for k in (all_rows[0].keys() if all_rows else []) if not k.endswith("curve")]
    rows_flat = []
    for r in all_rows:
        d = {k: r.get(k, np.nan) for k in flat_cols}
        for k in r:
            if k.startswith(("mix_", "side_")) and not k.endswith("curve"):
                d[k] = r[k]
        rows_flat.append(d)
    stops_df = pd.DataFrame(rows_flat)
    if len(stops_df):
        front = ["section", "name", "channel", "stop", "stop_str"]
        stops_df = stops_df[front + [c for c in stops_df.columns if c not in front]]
    save_csv(stops_df, out / "stops.csv")

    curve_rows = []
    for r in all_rows:
        for key, src in (("curve", "gtr"), ("mix_curve", "mix"), ("side_curve", "mix_side")):
            if key in r:
                for ms, v in zip(OFFSETS_MS, r[key]):
                    curve_rows.append(dict(section=r["section"], channel=r["channel"], stop=r["stop"],
                                           source=src, ms=int(ms), value=v))
    save_csv(pd.DataFrame(curve_rows), out / "curves.csv")

    # パートごとのまとめ
    sum_rows = []
    for s in secs:
        for ch in CHANNELS:
            rs = [r for r in all_rows if r["section"] == s["idx"] and r["channel"] == ch]
            row = dict(section=s["idx"], name=s["name"], start_str=fmt_time(s["start"]), channel=ch,
                       channel_ja=CH_JA[ch], n_stops=len(rs))
            for k in ("v80", "v150", "v250", "v350", "slope_100_300", "tail_LR_corr", "pre_LR_corr",
                      "tail_chroma_match", "knee_level", "slope_after"):
                row[k] = nan_median([r.get(k, np.nan) for r in rs])
            row["knee_ratio"] = float(np.mean([r["is_knee"] for r in rs])) if rs else np.nan
            mixed = [r for r in rs if "mix_curve" in r]
            row["n_mix_checked"] = len(mixed)
            row["n_all_stop"] = int(sum(r["all_stop"] for r in rs))
            for k in ("mix_v80", "mix_v150", "mix_v250", "mix_v350", "mix_slope_100_300", "mix_knee_level",
                      "mix_slope_after", "side_v80", "side_v250", "mix_tail_LR_corr", "mix_pre_LR_corr"):
                row[k] = nan_median([r.get(k, np.nan) for r in mixed])
            row["stop_times"] = " ".join(r["stop_str"] for r in rs[:12])
            sum_rows.append(row)
    sdf = pd.DataFrame(sum_rows)
    save_csv(sdf, out / "summary.csv")
    special = [dict(stop=fmt_time(r["stop"]), section=r["section"], channel=r["channel"],
                    all_stop=r["all_stop"], band_quiet=r["band_quiet"], is_last=r.get("is_last", False),
                    v80=r["v80"], mix_v80=r.get("mix_v80"), mix_v250=r.get("mix_v250"),
                    mix_knee_level=r.get("mix_knee_level"), mix_slope_after=r.get("mix_slope_after"),
                    side_v80=r.get("side_v80"))
               for r in all_rows if r["channel"] == "sum" and ("mix_curve" in r or r.get("is_last"))]
    save_json(out / "reverb.json", dict(reference=REFERENCE, sections=sdf.to_dict(orient="records"),
                                        mix_checked_stops=special))

    # 図：パートごとの減衰カーブ
    t = OFFSETS_MS
    cmap = plotting.plt.get_cmap("tab20")
    for ch in CHANNELS:
        fig, ax = plotting.plt.subplots(figsize=(10, 6))
        for k, s in enumerate(secs):
            rs = [r for r in all_rows if r["section"] == s["idx"] and r["channel"] == ch]
            if not rs:
                continue
            M = np.vstack([r["curve"] for r in rs])
            for row_ in M:
                ax.plot(t, row_, color=cmap(k % 20), lw=0.4, alpha=0.35)
            with np.errstate(all="ignore"):
                ax.plot(t, np.nanmedian(M, axis=0), color=cmap(k % 20), lw=1.8, label=f"{s['title']}（{len(rs)}）")
        ax.set_xlabel("止めてからの時間（ms）")
        ax.set_ylabel("直前の音を0dBとした音量（dB）")
        ax.set_ylim(-80, 5)
        ax.grid(alpha=0.3)
        ax.set_title(f"ギター{CH_JA[ch]}：止めた後の減衰（太線＝パートの中央値）")
        ax.legend(fontsize=6, loc="upper right")
        plotting.save(fig, out / f"decay_{ch}.png")

    mixed = [r for r in all_rows if r["channel"] == "sum" and "mix_curve" in r]
    if mixed:
        fig, axes = plotting.plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        for k, r in enumerate(mixed[:20]):
            lab = f"{r['stop_str']}（{'全員停止' if r['all_stop'] else '他の楽器ほぼ無し'}）"
            axes[0].plot(t, r["mix_curve"], color=cmap(k % 20), lw=1.2, label=lab)
            axes[0].plot(t, r["curve"], color=cmap(k % 20), lw=0.6, ls=":")
            axes[1].plot(t, r["side_curve"], color=cmap(k % 20), lw=1.2, label=lab)
        axes[0].set_title("原曲そのもの（実線）とギター分離音源（点線）")
        axes[1].set_title("原曲の左右の差の成分 L−R（原曲の直前の音＝0dB）")
        for ax in axes:
            ax.set_xlabel("止めてからの時間（ms）")
            ax.grid(alpha=0.3)
            ax.set_ylim(-80, 5)
        axes[0].set_ylabel("dB")
        axes[0].legend(fontsize=6)
        plotting.save(fig, out / "mix_decay.png")

    print(f"[reverb] 止めた場所 {len(all_rows)} 個（うち原曲でも確認 {len([r for r in all_rows if 'mix_curve' in r])} 個）")
    return sdf
