"""delay（ディレイの判定材料）

パートごと・左右別に：
1. 短い休符（直前の音より10dB以上下がり、0.06秒以上続く）を全部見つけ、止めてから80ms後の落ち幅を出す
2. 休符の中の音高プロファイル（CQT）を、t秒前（過去）と t秒後（未来）の音と比べた「過去−未来」の一致度の曲線
   （t = 0.15〜0.95秒）。過去とだけ強く一致する山が、ディレイの遅れ時間の候補
3. 音が切り替わるたびに、今の音にない音高（前の音の残り）を同じく過去と未来で比べる（休符が少ないパート用）
4. フィードバック込みのモデル（遅れD、繰り返しの強さ0／0.3／0.5）で当てはめ、一番一致するDを出す
5. 繰り返しが左右どちらに出るか（例：右の休符の時間に、左チャンネルに右の過去の音が出るか）
6. 参考：立ち上がりの自己相関（演奏の刻みと区別できないので主な根拠にしない）

判定（ディレイあり／なし）はしない。数値と図を残すだけ。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

from . import dsp, plotting
from .common import CH_JA, fmt_time, nan_median, save_csv, save_json

TAUS = np.round(np.arange(0.15, 0.9501, 0.01), 3)
DS = np.round(np.arange(0.15, 0.9501, 0.005), 3)
FBS = (0.0, 0.3, 0.5)
MAX_ECHO_S = 2.0
REST_DROP_DB = 10.0
REST_MIN_S = 0.06

REFERENCE = {
    "drop80": "止めて80ms後の落ち幅。乾いた音なら30dB以上落ちることが多い。10〜20dBしか落ちないなら、何かが残っている（ディレイ・リバーブ・他の音の混ざり・弾き方）",
    "rest_diff": "休符の音高が「t秒前に弾いた音」と「t秒後に弾く音」のどちらに似ているかの差。0付近＝区別なし。過去側にはっきりした山がある t がディレイの候補",
    "model": "フィードバック込みのモデルの一致度。fb=0（1回だけ）とfb=0.3／0.5（繰り返し）で、山の位置が揃っていれば候補の確度が上がる",
    "onset_acf": "立ち上がりの自己相関。8分や4分と重なる長さのディレイは演奏の刻みと区別できないので参考値",
}


# ---- プロファイルの比較 ---------------------------------------------------
def _unit_cols(M):
    n = np.linalg.norm(M, axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        U = np.where(n > 0, M / n, np.nan)
    return U


class Matcher:
    """target（余韻を探すチャンネル）の窓を、source（弾いた音のチャンネル）の過去・未来と比べる"""

    def __init__(self, C_target, C_source, hop_s):
        self.cst = np.concatenate([np.zeros((C_target.shape[0], 1)), np.cumsum(C_target, axis=1, dtype=np.float64)], 1)
        self.css = np.concatenate([np.zeros((C_source.shape[0], 1)), np.cumsum(C_source, axis=1, dtype=np.float64)], 1)
        self.n = C_source.shape[1]
        self.hop_s = hop_s

    def _shifted(self, a, b, shifts):
        """source の [a+s, b+s) の平均（列ごと）。範囲外は NaN"""
        aa, bb = a + shifts, b + shifts
        ok = (aa >= 0) & (bb <= self.n)
        out = np.full((self.css.shape[0], len(shifts)), np.nan)
        if ok.any():
            out[:, ok] = (self.css[:, bb[ok]] - self.css[:, aa[ok]]) / (b - a)
        return out

    def target(self, a, b):
        return (self.cst[:, b] - self.cst[:, a]) / (b - a)

    def curves(self, windows, taus):
        """windows: (a, b, mask) のリスト。返り値：過去の一致度、未来の一致度（窓 × tau）"""
        sh = np.round(taus / self.hop_s).astype(int)
        past = np.full((len(windows), len(taus)), np.nan)
        fut = np.full((len(windows), len(taus)), np.nan)
        for w, (a, b, mask) in enumerate(windows):
            R = self.target(a, b) * mask
            nr = np.linalg.norm(R)
            if nr <= 0:
                continue
            R = R / nr
            P = _unit_cols(self._shifted(a, b, -sh) * mask[:, None])
            F = _unit_cols(self._shifted(a, b, sh) * mask[:, None])
            past[w] = R @ P
            fut[w] = R @ F
        return past, fut

    def model(self, windows, ds, fbs):
        """フィードバック込みのモデル：E = Σ fb^(k-1) × (k×D 前の音)。未来側も同じ形で作って差を取る"""
        kmax = int(np.floor(MAX_ECHO_S / ds.min()))
        out = {fb: np.full((len(windows), len(ds)), np.nan) for fb in fbs}
        for w, (a, b, mask) in enumerate(windows):
            R = self.target(a, b) * mask
            nr = np.linalg.norm(R)
            if nr <= 0:
                continue
            R = R / nr
            Ps, Fs = [], []
            for k in range(1, kmax + 1):
                sh = np.round(k * ds / self.hop_s).astype(int)
                P = _unit_cols(self._shifted(a, b, -sh) * mask[:, None])
                F = _unit_cols(self._shifted(a, b, sh) * mask[:, None])
                too_far = k * ds > MAX_ECHO_S
                P[:, too_far] = 0
                F[:, too_far] = 0
                Ps.append(P)
                Fs.append(F)
            for fb in fbs:
                E = np.zeros_like(Ps[0])
                Fu = np.zeros_like(Fs[0])
                for k, (P, F) in enumerate(zip(Ps, Fs)):
                    wgt = fb ** k if fb > 0 else (1.0 if k == 0 else 0.0)
                    if wgt == 0:
                        continue
                    if k == 0:
                        E += wgt * P
                        Fu += wgt * F
                    else:
                        E += wgt * np.nan_to_num(P)
                        Fu += wgt * np.nan_to_num(F)
                out[fb][w] = R @ _unit_cols(E) - R @ _unit_cols(Fu)
        return out


def _mean_curve(M):
    with np.errstate(all="ignore"):
        ok = np.isfinite(M)
        cnt = ok.sum(axis=0)
        s = np.where(ok, M, 0).sum(axis=0)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan), int(np.max(cnt) if len(cnt) else 0)


# ---- 休符と音の切り替わり ------------------------------------------------
def detect_rests(ctx, ch, t0, t1):
    env = ctx.env("gtr", ch)
    hop = ctx.env_hop_s
    min_ref = ctx.active_threshold("gtr") + 10
    rests = dsp.find_rests(env, hop, ctx.env_index(t0), ctx.env_index(t1), drop_db=REST_DROP_DB,
                           min_len=REST_MIN_S, min_ref=min_ref)
    rows = []
    for r in rests:
        stop_t, end_t = r["stop"] * hop, r["end"] * hop
        row = dict(stop=stop_t, start=r["start"] * hop, end=end_t, rest_len=end_t - stop_t, ref_db=r["ref"])
        for ms in (50, 80):
            i = r["stop"] + int(round(ms / 1000 / hop))
            row[f"drop{ms}"] = float(r["ref"] - env[i]) if (i * hop) < end_t - 0.005 else np.nan
        rows.append(row)
    return rows


def rest_windows(ctx, rests):
    out = []
    n = ctx.cqt("gtr", "sum").shape[1]
    for r in rests:
        a = ctx.frame_index(r["stop"] + 0.03)
        b = max(a + 2, ctx.frame_index(r["end"] - 0.005))
        if b <= n:
            out.append((a, b, np.ones(72)))
    return out


def note_change_windows(ctx, ch, t0, t1):
    """音の切り替わり：次の音の直前の窓で、今の音にない音高だけを残す（mask）"""
    C = ctx.cqt("gtr", ch)
    on = ctx.onsets("gtr", ch)
    on = on[(on * ctx.hop_s >= t0) & (on * ctx.hop_s < t1)]
    act = ctx.active_frames("gtr", ch)
    min_gap = int(round(0.15 / ctx.hop_s))
    out = []
    for o, o2 in zip(on[:-1], on[1:]):
        if o2 - o < min_gap or not act[o:o2].mean() > 0.5:
            continue
        N = C[:, o + 1:o + 4].mean(axis=1)
        if N.max() <= 0:
            continue
        mask = (maximum_filter1d(N, 3) < 0.3 * N.max()).astype(float)
        a, b = max(o + 5, o2 - 8), o2 - 1
        if b - a < 2:
            continue
        W = C[:, a:b].mean(axis=1)
        if np.linalg.norm(W * mask) < 0.1 * np.linalg.norm(W):
            continue
        out.append((a, b, mask))
    return out


def onset_acf(ctx, ch, t0, t1):
    e = ctx.onset_env("gtr", ch)[ctx.frame_index(t0):ctx.frame_index(t1)]
    if len(e) < 50:
        return None, []
    r = dsp.acf(e)
    lags = np.arange(len(r)) * ctx.hop_s
    sel = (lags >= 0.15) & (lags <= 0.95)
    return (lags[sel], r[sel]), dsp.top_peaks(lags, r, n=3, min_x=0.15, max_x=0.95)


# ---- 実行 ----------------------------------------------------------------
def analyze_range(ctx, t0, t1, channels=("L", "R", "sum")):
    """区間 [t0, t1) を調べる（check からも使う）"""
    res = {}
    for ch in channels:
        rests = detect_rests(ctx, ch, t0, t1)
        C = ctx.cqt("gtr", ch)
        m_same = Matcher(C, C, ctx.hop_s)
        rw = rest_windows(ctx, rests)
        nw = note_change_windows(ctx, ch, t0, t1)
        d = dict(rests=rests, n_rests=len(rests), n_note_changes=len(nw))
        d80 = [r["drop80"] for r in rests]
        d50 = [r["drop50"] for r in rests]
        d["drop80_median"] = nan_median(d80)
        d["drop80_n"] = int(np.sum(np.isfinite(d80)))
        d["drop80_p25"] = float(np.nanpercentile(d80, 25)) if d["drop80_n"] else np.nan
        d["drop80_p75"] = float(np.nanpercentile(d80, 75)) if d["drop80_n"] else np.nan
        d["drop50_median"] = nan_median(d50)
        curves = {}
        if rw:
            p, f = m_same.curves(rw, TAUS)
            curves["rest_past"], _ = _mean_curve(p)
            curves["rest_future"], _ = _mean_curve(f)
            curves["rest_diff"], _ = _mean_curve(p - f)
        if nw:
            p, f = m_same.curves(nw, TAUS)
            curves["note_diff"], _ = _mean_curve(p - f)
        if ch in ("L", "R") and rw:
            other = "R" if ch == "L" else "L"
            m_cross = Matcher(ctx.cqt("gtr", other), C, ctx.hop_s)
            p, f = m_cross.curves(rw, TAUS)
            curves["cross_diff"], _ = _mean_curve(p - f)
        models = {}
        mr = m_same.model(rw, DS, FBS) if rw else None
        mn = m_same.model(nw, DS, FBS) if nw else None
        for fb in FBS:
            parts = []
            if mr is not None:
                parts.append(_mean_curve(mr[fb])[0])
            if mn is not None:
                parts.append(_mean_curve(mn[fb])[0])
            if parts:
                with np.errstate(all="ignore"):
                    models[fb] = np.nanmean(np.vstack(parts), axis=0)
        d["curves"] = curves
        d["models"] = models
        for key in ("rest_diff", "note_diff", "cross_diff"):
            if key in curves:
                d[f"{key}_peaks"] = dsp.top_peaks(TAUS, curves[key], n=3)
        best = {}
        for fb, mc in models.items():
            if np.any(np.isfinite(mc)):
                i = int(np.nanargmax(mc))
                best[fb] = (float(DS[i]), float(mc[i]))
        d["model_best"] = best
        # 左右：一番当てはまったDで、もう一方のチャンネルに同じ音が出ているか
        if "cross_diff" in curves and best.get(0.0):
            D = best[0.0][0]
            j = int(np.argmin(np.abs(TAUS - D)))
            d["same_at_D"] = float(curves["rest_diff"][j]) if "rest_diff" in curves else np.nan
            d["cross_at_D"] = float(curves["cross_diff"][j])
        acf_curve, acf_peaks = onset_acf(ctx, ch, t0, t1)
        d["onset_acf"] = acf_curve
        d["onset_acf_peaks"] = acf_peaks
        res[ch] = d
    return res


def _fmt_peaks(pk):
    return " / ".join(f"{x * 1000:.0f}ms({y:+.3f})" for x, y in pk) if pk else ""


def plot_range(ctx, res, title, path):
    chans = [c for c in ("L", "R", "sum") if c in res]
    fig, axes = plotting.plt.subplots(len(chans), 2, figsize=(14, 3.2 * len(chans)), squeeze=False,
                                      gridspec_kw=dict(hspace=0.6))
    for row, ch in enumerate(chans):
        d = res[ch]
        ax = axes[row, 0]
        cv = d["curves"]
        styles = [("rest_diff", "休符：過去−未来", "tab:red", "-"),
                  ("note_diff", "音の切り替わり：過去−未来", "tab:blue", "-"),
                  ("cross_diff", "反対チャンネルの同じ時間：過去−未来", "tab:purple", "--"),
                  ("rest_past", "休符：過去との一致", "tab:red", ":"),
                  ("rest_future", "休符：未来との一致", "0.5", ":")]
        for key, lab, c, ls in styles:
            if key in cv:
                y = cv[key]
                if key in ("rest_past", "rest_future"):
                    y = y - np.nanmean(y)  # 形だけ見るために平均を引く
                    lab += "（平均を引いた形）"
                ax.plot(TAUS, y, ls, color=c, lw=1.2 if ls == "-" else 0.8, label=lab)
        if d.get("onset_acf") is not None:
            lg, r = d["onset_acf"]
            ax2 = ax.twinx()
            ax2.plot(lg, r, color="0.7", lw=0.7)
            ax2.set_ylabel("立ち上がり自己相関（参考）", fontsize=6, color="0.5")
            ax2.tick_params(labelsize=6, colors="0.5")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(TAUS[0], TAUS[-1])
        plotting.note_value_lines(ax, ctx)
        ax.set_xlabel("t（秒）")
        ax.set_title(f"ギター{CH_JA[ch]}：休符{d['n_rests']}個（80ms後の落ち幅 中央値 "
                     f"{d['drop80_median']:.1f}dB, n={d['drop80_n']}）・切り替わり{d['n_note_changes']}個",
                     fontsize=8, loc="left", pad=18)
        ax.legend(fontsize=6, loc="lower right")
        ax = axes[row, 1]
        for fb, c in zip(FBS, ("tab:green", "tab:orange", "tab:brown")):
            if fb in d["models"]:
                b = d["model_best"].get(fb)
                ax.plot(DS, d["models"][fb], color=c, lw=1.1,
                        label=f"fb={fb}" + (f"  最大 {b[0] * 1000:.0f}ms" if b else ""))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(DS[0], DS[-1])
        plotting.note_value_lines(ax, ctx)
        ax.set_xlabel("遅れ時間 D（秒）")
        ax.set_title("フィードバック込みモデルの一致度（過去−未来）", fontsize=8, loc="left", pad=18)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=10, y=1.0)
    plotting.save(fig, path)


def run(ctx):
    out = ctx.subdir("delay")
    secs = ctx.sections()
    q = ctx.beat_s
    rest_rows, curve_rows, sum_rows = [], [], []
    js = {"reference": REFERENCE, "note_lengths_ms": {"4分": q * 1000, "付点8分": 0.75 * q * 1000,
                                                      "8分": 0.5 * q * 1000, "付点4分": 1.5 * q * 1000},
          "sections": []}
    for s in secs:
        res = analyze_range(ctx, s["start"], s["end"])
        sj = dict(section=s["idx"], name=s["name"], start=s["start"], end=s["end"], channels={})
        for ch, d in res.items():
            for r in d["rests"]:
                rest_rows.append(dict(section=s["idx"], name=s["name"], channel=ch, stop=r["stop"],
                                      stop_str=fmt_time(r["stop"]), rest_end=r["end"], rest_len=r["rest_len"],
                                      ref_db=r["ref_db"], drop50=r["drop50"], drop80=r["drop80"]))
            for key, y in d["curves"].items():
                for t, v in zip(TAUS, y):
                    curve_rows.append(dict(section=s["idx"], channel=ch, method=key, t=t, value=v))
            for fb, y in d["models"].items():
                for t, v in zip(DS, y):
                    curve_rows.append(dict(section=s["idx"], channel=ch, method=f"model_fb{fb}", t=t, value=v))
            row = dict(section=s["idx"], name=s["name"], start_str=fmt_time(s["start"]), channel=ch,
                       channel_ja=CH_JA[ch], n_rests=d["n_rests"], drop80_n=d["drop80_n"],
                       drop80_median=d["drop80_median"], drop80_p25=d["drop80_p25"], drop80_p75=d["drop80_p75"],
                       drop50_median=d["drop50_median"], n_note_changes=d["n_note_changes"],
                       rest_diff_peaks=_fmt_peaks(d.get("rest_diff_peaks")),
                       note_diff_peaks=_fmt_peaks(d.get("note_diff_peaks")),
                       cross_diff_peaks=_fmt_peaks(d.get("cross_diff_peaks")),
                       onset_acf_peaks_ref=_fmt_peaks(d.get("onset_acf_peaks")),
                       same_at_D=d.get("same_at_D", np.nan), cross_at_D=d.get("cross_at_D", np.nan))
            for fb in FBS:
                b = d["model_best"].get(fb)
                row[f"best_D_fb{fb}_ms"] = b[0] * 1000 if b else np.nan
                row[f"best_score_fb{fb}"] = b[1] if b else np.nan
            sum_rows.append(row)
            sj["channels"][ch] = {k: v for k, v in row.items() if k not in ("section", "name", "start_str", "channel")}
        js["sections"].append(sj)
        plot_range(ctx, res, f"ディレイの判定材料：{s['title']}", out / f"{s['label']}.png")
    save_csv(pd.DataFrame(rest_rows), out / "rests.csv")
    save_csv(pd.DataFrame(curve_rows), out / "curves.csv")
    sdf = pd.DataFrame(sum_rows)
    save_csv(sdf, out / "summary.csv")
    save_json(out / "delay.json", js)

    # 全体図：パートごとの80ms後の落ち幅
    fig, ax = plotting.plt.subplots(figsize=(12, 0.4 * len(secs) + 1.5))
    yy = np.arange(len(secs))
    for ch, c, off in (("L", "tab:blue", -0.2), ("R", "tab:red", 0.0), ("sum", "k", 0.2)):
        d = sdf[sdf.channel == ch].set_index("section").reindex([s["idx"] for s in secs])
        ax.barh(yy + off, d.drop80_median, height=0.2, color=c, label=f"ギター{CH_JA[ch]}")
        ax.errorbar(d.drop80_median, yy + off, xerr=[d.drop80_median - d.drop80_p25, d.drop80_p75 - d.drop80_median],
                    fmt="none", ecolor="0.4", lw=0.6)
    ax.set_yticks(yy)
    ax.set_yticklabels([s["title"] for s in secs], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("止めてから80ms後の落ち幅（dB、中央値と25〜75%）  小さいほど音が残っている")
    ax.legend(fontsize=7)
    plotting.save(fig, out / "drop80.png")
    print(f"[delay] {len(secs)} 区間、休符 {len(rest_rows)} 個")
    return sdf
