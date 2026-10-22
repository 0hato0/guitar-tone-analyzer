"""structure（構成と小節）：拍の追跡、ギターの最初の音を基準にした小節、小節ごとの表、構成の切れ目の候補。

sections.csv の扱い：
- results/sections.csv がなければ、自動で出した区切りをそのまま書く
- すでにあれば（手で直した場合も含めて）上書きしない。自動の候補は structure/sections_auto.csv に毎回出す
- 作り直したいときは --reset-sections を付けて実行する
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import librosa
from scipy import signal

from . import plotting
from .common import (ENV_HOP, HOP, NOTE_NAMES, corr, fmt_time, parse_time,
                     rms_db, save_csv, save_json)

MIN_SECTION_BARS = 2
KERNEL_HALF = 4  # 新しさ（ノベルティ）を見る幅：前後4小節


# ---- 拍 -----------------------------------------------------------------
def guitar_first_onset(ctx) -> float:
    env = ctx.env("gtr", "sum")
    thr = ctx.ref_level("gtr") - 30.0
    hop_s = ctx.env_hop_s
    n_hold = int(0.2 / hop_s)
    above = env > thr
    t_first = None
    for i in np.flatnonzero(above):
        if above[i:i + n_hold].mean() >= 0.5:
            t_first = i * hop_s
            break
    if t_first is None:
        return 0.0
    on = librosa.onset.onset_detect(onset_envelope=ctx.onset_env("gtr", "sum"), sr=ctx.sr,
                                    hop_length=HOP, backtrack=True, units="time")
    near = on[(on <= t_first + 0.05) & (on >= t_first - 0.3)]
    return float(near[-1]) if len(near) else float(t_first)


def compute_beats(ctx) -> dict:
    env = ctx.onset_env("mix", "sum")
    tempo_est, frames = librosa.beat.beat_track(onset_envelope=env, sr=ctx.sr, hop_length=HOP)
    tempo_est = float(np.atleast_1d(tempo_est)[0])
    bt = librosa.frames_to_time(frames, sr=ctx.sr, hop_length=HOP)
    if len(bt) < 4:
        ibi = 60.0 / (tempo_est or 120.0)
        bt = np.arange(0, ctx.duration, ibi)
    ibi = float(np.median(np.diff(bt)))
    tempo = 60.0 / ibi
    pre = []
    t = bt[0] - ibi
    while t >= -1e-6:
        pre.append(max(t, 0.0))
        t -= ibi
    post = []
    t = bt[-1] + ibi
    while t < ctx.duration:
        post.append(t)
        t += ibi
    beats = np.concatenate([np.array(pre[::-1]), bt, np.array(post)])
    extrap = np.concatenate([np.ones(len(pre), bool), np.zeros(len(bt), bool), np.ones(len(post), bool)])

    g_first = guitar_first_onset(ctx)
    if ctx.settings.get("downbeat"):
        anchor = parse_time(ctx.settings["downbeat"])
        src = "手動指定（results/settings.json の downbeat）"
    else:
        anchor = g_first
        src = "ギターの最初の音"
    bpb = ctx.beats_per_bar
    j = int(np.argmin(np.abs(beats - anchor)))
    idx = np.arange(len(beats))
    beat_in_bar = (idx - j) % bpb + 1
    bar_raw = np.floor((idx - j) / bpb).astype(int)
    bar = bar_raw - bar_raw[0] + 1
    info = dict(tempo=tempo, tempo_librosa=tempo_est, beats=beats, beat_in_bar=beat_in_bar, bar=bar,
                extrapolated=extrap, anchor=anchor, anchor_source=src, guitar_first_onset=g_first,
                anchor_bar=int(bar[j]), beats_per_bar=bpb, duration_s=ctx.duration)

    out = ctx.subdir("structure")
    save_csv(pd.DataFrame(dict(time=beats, time_str=[fmt_time(b) for b in beats], bar=bar,
                               beat_in_bar=beat_in_bar, extrapolated=extrap)), out / "beats.csv")
    save_json(out / "beat_meta.json", {k: v for k, v in info.items()
                                       if k not in ("beats", "beat_in_bar", "bar", "extrapolated")})
    return info


# ---- 小節ごとの表 --------------------------------------------------------
def bar_table(ctx) -> pd.DataFrame:
    bi = ctx.beat_info()
    beats, bars = bi["beats"], bi["bar"]
    starts = []
    for b in np.unique(bars):
        starts.append((int(b), float(beats[bars == b][0])))
    starts[0] = (starts[0][0], 0.0)
    sr = ctx.sr
    mix, gtr, oth = ctx.sig("mix", "sum"), ctx.sig("gtr", "sum"), ctx.sig("oth", "sum")
    gl, gr = ctx.sig("gtr", "L"), ctx.sig("gtr", "R")
    chroma = ctx.chroma("gtr", "sum")
    chroma_mix = ctx.chroma("mix", "sum")
    act = ctx.active_frames("gtr", "sum")
    rows = []
    for k, (b, t0) in enumerate(starts):
        t1 = starts[k + 1][1] if k + 1 < len(starts) else ctx.duration
        a, e = int(t0 * sr), int(t1 * sr)
        fa, fe = ctx.frame_index(t0), max(ctx.frame_index(t1), ctx.frame_index(t0) + 1)
        ch = chroma[:, fa:fe][:, act[fa:fe]] if act[fa:fe].any() else np.zeros((12, 1))
        cvec = ch.mean(axis=1) if ch.size else np.zeros(12)
        cvec = cvec / cvec.max() if cvec.max() > 0 else cvec
        cmix = chroma_mix[:, fa:fe].mean(axis=1)
        cmix = cmix / cmix.max() if cmix.max() > 0 else cmix
        L, R = rms_db(gl[a:e]), rms_db(gr[a:e])
        row = dict(bar=b, start=t0, end=t1, start_str=fmt_time(t0),
                   mix_db=rms_db(mix[a:e]), gtr_db=rms_db(gtr[a:e]), oth_db=rms_db(oth[a:e]),
                   gtr_L_db=L, gtr_R_db=R, gtr_R_minus_L_db=R - L, gtr_LR_corr=corr(gl[a:e], gr[a:e]),
                   gtr_active=float(act[fa:fe].mean()) if fe > fa else 0.0,
                   chord_guess=chord_guess(cvec) if cvec.max() > 0 else "",
                   top_pitch=" ".join(NOTE_NAMES[i] for i in np.argsort(-cvec)[:3]) if cvec.max() > 0 else "")
        for i, n in enumerate(NOTE_NAMES):
            row[f"chroma_{n}"] = float(cvec[i])
        for i, n in enumerate(NOTE_NAMES):
            row[f"mixchroma_{n}"] = float(cmix[i])
        rows.append(row)
    return pd.DataFrame(rows)


_TEMPLATES = {"": [0, 4, 7], "m": [0, 3, 7], "5": [0, 7], "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10]}


def chord_guess(cvec) -> str:
    """クロマに一番近い和音（目安。歪みの倍音で5度が強く出るので参考程度）"""
    best, name = -1, ""
    for root in range(12):
        for suf, iv in _TEMPLATES.items():
            t = np.zeros(12)
            t[[(root + i) % 12 for i in iv]] = 1
            s = float(np.dot(cvec, t) / (np.linalg.norm(cvec) * np.linalg.norm(t) + 1e-9))
            if s > best + 1e-6:
                best, name = s, NOTE_NAMES[root] + suf
    return name


# ---- 構成の切れ目 --------------------------------------------------------
def _z(X):
    X = np.asarray(X, float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    return (X - X.mean(axis=0)) / sd


def novelty_boundaries(ctx, bars: pd.DataFrame):
    sr = ctx.sr
    mf_mix = librosa.feature.mfcc(y=ctx.sig("mix", "sum"), sr=sr, hop_length=HOP, n_mfcc=13)
    mf_gtr = librosa.feature.mfcc(y=ctx.sig("gtr", "sum"), sr=sr, hop_length=HOP, n_mfcc=13)
    fm, fg = [], []
    for _, r in bars.iterrows():
        a, e = ctx.frame_index(r.start), max(ctx.frame_index(r.end), ctx.frame_index(r.start) + 1)
        fm.append(mf_mix[:, a:e].mean(axis=1))
        fg.append(mf_gtr[:, a:e].mean(axis=1))
    lv = bars[["mix_db", "gtr_db", "oth_db", "gtr_L_db", "gtr_R_db"]].to_numpy(float)
    lv = np.maximum(lv, -80)
    lc = np.nan_to_num(bars[["gtr_LR_corr"]].to_numpy(float), nan=0.0)
    groups = [
        (bars[[f"chroma_{n}" for n in NOTE_NAMES]].to_numpy(float), 1.0),
        (bars[[f"mixchroma_{n}" for n in NOTE_NAMES]].to_numpy(float), 1.0),
        (np.array(fm), 1.0),
        (np.array(fg), 1.0),
        (np.hstack([lv, lc]), 1.5),
    ]
    F = np.hstack([_z(g) * w / np.sqrt(g.shape[1]) for g, w in groups])
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    S = Fn @ Fn.T
    n = len(S)
    L = KERNEL_HALF
    g = signal.windows.gaussian(2 * L, std=L / 1.5)
    K = np.outer(g, g) * np.block([[np.ones((L, L)), -np.ones((L, L))], [-np.ones((L, L)), np.ones((L, L))]])
    Sp = np.pad(S, L, mode="edge")
    nov = np.zeros(n)
    for i in range(n):
        nov[i] = float(np.sum(K * Sp[i:i + 2 * L, i:i + 2 * L]))
    nov = np.maximum(nov, 0)
    if nov.max() > 0:
        nov = nov / nov.max()
    peaks, _ = signal.find_peaks(nov, distance=4, prominence=0.1)
    cands = {int(p): (float(nov[p]), "特徴の変化") for p in peaks if p > 0}

    # ギターが入る小節と、ギターの音量が大きく変わる小節も候補に足す
    gdb = np.maximum(bars["gtr_db"].to_numpy(float), -90)
    g_first = ctx.beat_info()["guitar_first_onset"]
    gi = int(np.searchsorted(bars["start"].to_numpy(), g_first + 1e-3) - 1)
    if gi > 0:
        cands[gi] = (np.inf, "ギターが入る")
    for i in range(1, n):
        if abs(gdb[i] - gdb[i - 1]) >= 10 and not any(abs(i - c) < 2 for c in cands):
            cands[i] = (0.5, f"ギターの音量が{gdb[i] - gdb[i - 1]:+.0f}dB")
    # 近すぎる候補は点数の高い方を残す
    keep = []
    for i in sorted(cands, key=lambda c: -cands[c][0]):
        if all(abs(i - k) >= MIN_SECTION_BARS for k in keep):
            keep.append(i)
    keep = sorted(keep)
    return S, nov, keep, cands


def run(ctx):
    out = ctx.subdir("structure")
    bi = ctx.beat_info(force=True)
    bars = bar_table(ctx)
    S, nov, bounds, cands = novelty_boundaries(ctx, bars)

    starts = [0] + [b for b in bounds if b > 0]
    rows = []
    for k, i in enumerate(starts):
        j = starts[k + 1] if k + 1 < len(starts) else len(bars)
        t0 = 0.0 if k == 0 else float(bars.start.iloc[i])
        t1 = float(bars.start.iloc[j]) if j < len(bars) else ctx.duration
        sc, why = cands.get(i, (np.nan, "曲の頭"))
        rows.append(dict(start=fmt_time(t0), end=fmt_time(t1), name="", bars=j - i,
                         start_bar=int(bars.bar.iloc[i]), score=sc if np.isfinite(sc) else np.nan, reason=why))
    auto = pd.DataFrame(rows)
    save_csv(auto, out / "sections_auto.csv")
    sec_path = ctx.results / "sections.csv"
    if ctx.reset_sections or not sec_path.exists():
        save_csv(auto[["start", "end", "name", "bars", "start_bar"]], sec_path)
        print(f"[structure] sections.csv を作りました（{len(auto)} 区間、パート名は空欄）")
    else:
        print("[structure] 既存の sections.csv を使います（作り直すときは --reset-sections）")
    ctx.invalidate_sections()
    secs = ctx.sections()

    # 小節表に区間番号を付けて保存
    sec_idx = []
    for t in bars.start:
        s = ctx.section_of(float(t) + 1e-3)
        sec_idx.append(s["idx"] if s else np.nan)
    bars.insert(1, "section", sec_idx)
    save_csv(bars, out / "bars.csv")

    # 区間ごとの小節数（sections.csv の時間から数え直す）
    db_ = ctx.downbeats()
    sec_rows = []
    for s in secs:
        nb = int(np.sum((db_ >= s["start"] - 0.05) & (db_ < s["end"] - 0.05)))
        sec_rows.append(dict(idx=s["idx"], name=s["name"], start=s["start"], end=s["end"],
                             start_str=fmt_time(s["start"]), end_str=fmt_time(s["end"]),
                             length_s=s["end"] - s["start"], bars=nb))
    sec_df = pd.DataFrame(sec_rows)
    save_csv(sec_df, out / "sections_detail.csv")

    res = dict(tempo=bi["tempo"], beats_per_bar=bi["beats_per_bar"],
               guitar_first_onset=bi["guitar_first_onset"], guitar_first_onset_str=fmt_time(bi["guitar_first_onset"]),
               anchor=bi["anchor"], anchor_source=bi["anchor_source"], anchor_bar=bi["anchor_bar"],
               n_bars=int(len(bars)), sections=sec_rows,
               auto_boundaries=[dict(bar=int(bars.bar.iloc[i]), time=fmt_time(bars.start.iloc[i]),
                                     reason=cands[i][1]) for i in bounds])
    save_json(out / "structure.json", res)

    # ---- 図：自己類似と新しさ ----
    fig, axes = plotting.plt.subplots(2, 1, figsize=(10, 12), gridspec_kw=dict(height_ratios=[4, 1]))
    ax = axes[0]
    ax.imshow(S, origin="lower", cmap="magma", vmin=-0.2, vmax=1, aspect="equal")
    for b in bounds:
        ax.axhline(b - 0.5, color="cyan", lw=0.6)
        ax.axvline(b - 0.5, color="cyan", lw=0.6)
    tick = list(range(0, len(bars), 8))
    ax.set_xticks(tick)
    ax.set_xticklabels([f"{bars.bar.iloc[i]}\n{fmt_time(bars.start.iloc[i], 0)}" for i in tick], fontsize=6)
    ax.set_yticks(tick)
    ax.set_yticklabels([str(bars.bar.iloc[i]) for i in tick], fontsize=6)
    ax.set_title("小節ごとの自己類似（明るい＝似ている）と自動の区切り（水色）")
    ax = axes[1]
    ax.plot(np.arange(len(nov)), nov, color="k", lw=1)
    for b in bounds:
        ax.axvline(b, color="tab:red", lw=0.8)
    ax.set_xlim(0, len(nov))
    ax.set_xticks(tick)
    ax.set_xticklabels([str(bars.bar.iloc[i]) for i in tick], fontsize=6)
    ax.set_xlabel("小節")
    ax.set_ylabel("新しさ")
    plotting.save(fig, out / "ssm.png")

    # ---- 図：小節ごとの音量・左右・クロマ ----
    fig, axes = plotting.plt.subplots(4, 1, figsize=(14, 11), sharex=True,
                                      gridspec_kw=dict(height_ratios=[2, 1.5, 1, 2], hspace=0.25))
    x = bars.start.to_numpy()
    ax = axes[0]
    for col, lab, c in (("mix_db", "原曲", "k"), ("gtr_db", "ギター", "tab:red"), ("oth_db", "他の楽器", "tab:blue")):
        ax.step(x, np.maximum(bars[col], -80), where="post", label=lab, color=c, lw=1)
    ax.set_ylabel("音量 dB")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("小節ごとの音量・ギターの左右・ギターのクロマ")
    ax = axes[1]
    ax.step(x, np.maximum(bars.gtr_L_db, -80), where="post", label="ギター左", color="tab:blue", lw=1)
    ax.step(x, np.maximum(bars.gtr_R_db, -80), where="post", label="ギター右", color="tab:red", lw=1)
    ax.set_ylabel("dB")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[2]
    ax.step(x, bars.gtr_LR_corr, where="post", color="k", lw=1)
    ax.set_ylim(-0.2, 1.05)
    ax.set_ylabel("左右相関")
    ax = axes[3]
    C = bars[[f"chroma_{n}" for n in NOTE_NAMES]].to_numpy(float).T
    edges = np.append(x, bars.end.iloc[-1])
    ax.pcolormesh(edges, np.arange(13) - 0.5, C, cmap="Greys", shading="flat")
    ax.set_yticks(range(12))
    ax.set_yticklabels(NOTE_NAMES, fontsize=7)
    ax.set_ylabel("クロマ")
    for a in axes:
        plotting.draw_grid(a, ctx, 0, ctx.duration, labels=(a is axes[0]))
    plotting.save(fig, out / "bars.png")

    print(f"[structure] テンポ {bi['tempo']:.1f} / ギターの最初の音 {fmt_time(bi['guitar_first_onset'])}"
          f"（第{bi['anchor_bar']}小節の頭）/ {len(bars)} 小節 / {len(secs)} 区間")
    return res
