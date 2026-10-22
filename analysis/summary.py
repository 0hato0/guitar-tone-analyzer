"""summary.md：各モジュールの出力（CSV・JSON）から、主な数値を人間が読める表にまとめる。"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from .common import CH_JA, GEQ7_LABELS, fmt_time, load_json, read_csv


def _f(v, fmt="{:.1f}", empty="-"):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return empty
        if isinstance(v, str):
            return v if v else empty
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    return out


def _read(path):
    return read_csv(path) if path.exists() else None


def build(ctx):
    R = ctx.results
    L = [f"# 解析のまとめ：{ctx.song_name}", "",
         f"作成：{_dt.datetime.now():%Y-%m-%d %H:%M}　／　数値はすべて推定（分離音源には他の楽器が混ざる）。"
         "しきい値は参考値で、「あり／なし」の判断はここではしていない。", ""]
    secs = ctx.sections()

    # ---- 基本情報 ----
    b = load_json(R / "basic" / "basic.json")
    if b:
        o = b["original"]
        L += ["## 基本情報", ""]
        L += _table(["項目", "値"], [
            ["原曲", f"{b['files']['原曲']}（{o['原曲']['sr']}Hz, {o['原曲']['channels']}ch, {fmt_time(o['原曲']['duration_s'], 1)}）"],
            ["ギター分離音源", f"{b['files']['ギター']}（{o['ギター']['sr']}Hz, {o['ギター']['channels']}ch, {fmt_time(o['ギター']['duration_s'], 1)}）"],
            ["解析", f"{b['analysis_sr']}Hz に揃え、長さ {b['duration']}（短い方に合わせた）"],
            ["時間のずれ", f"{b['offset']['ms']:+.1f} ms（{'補正した' if b['offset']['corrected'] else '補正なし'}"
                         f"{'、位相反転' if b['offset']['polarity_inverted'] else ''}）"],
            ["ギターの音量倍率（原曲に対する最小二乗）", f"{b['guitar_gain_ls']:.2f}（他の楽器を作るときは {b['guitar_gain_used']:.2f} を使用）"],
            ["テンポ（曲全体）", f"{b['tempo_bpm']:.1f} BPM"],
            ["テンポ（20秒ごと、拍の間隔から）", f"{_f(b['tempo_windows_range'][0])}〜{_f(b['tempo_windows_range'][1])} BPM"],
            ["チューニング", f"{b['tuning_cents']:+.1f} セント（A={b['tuning_a4_hz']:.1f}Hz）。原曲全体では {_f(b['tuning_cents_mix'], '{:+.1f}')} セント"],
            ["最低音（推定）", f"合計 {b['lowest_note'].get('sum')}／左 {b['lowest_note'].get('L')}／右 {b['lowest_note'].get('R')}"],
        ])
        tw = _read(R / "basic" / "tempo_windows.csv")
        if tw is not None and tw["note"].fillna("").str.len().sum() > 0:
            L += ["", "テンポの注意（20秒ごと）："]
            for _, r in tw.iterrows():
                if isinstance(r["note"], str) and r["note"]:
                    L.append(f"- {r['start_str']}〜：{r['note']}")
        L.append("")

    # ---- 構成 ----
    st = load_json(R / "structure" / "structure.json")
    feat = _read(R / "features" / "features.csv")
    if st:
        L += ["## 構成", "",
              f"拍子 {st['beats_per_bar']}/4。ギターの最初の音 {st['guitar_first_onset_str']}。小節の頭の基準：{st['anchor_source']}"
              f"（第{st['anchor_bar']}小節）。区切りは results/sections.csv（手で直したらそれが使われる）。", ""]
        rows = []
        for s in st["sections"]:
            fs = feat[(feat.section == s["idx"]) & (feat.channel == "sum")] if feat is not None else None
            r = fs.iloc[0] if fs is not None and len(fs) else None
            rows.append([s["idx"], s["name"] or "", f"{s['start_str']}〜{s['end_str']}", f"{s['length_s']:.1f}",
                         s["bars"],
                         _f(r["level_db"]) if r is not None else "-",
                         _f(r["vs_other_db"], "{:+.1f}") if r is not None else "-",
                         _f(r["R_minus_L_db"], "{:+.1f}") if r is not None else "-",
                         _f(r["LR_corr"], "{:.2f}") if r is not None else "-"])
        L += _table(["#", "パート名", "時間", "秒", "小節", "ギター音量 dB", "他の楽器との差", "右−左 dB", "左右相関"], rows)
        L.append("")

    # ---- パートごとの数値 ----
    if feat is not None:
        L += ["## パートごとの数値（ギター分離音源）", "",
              "GEQ7 は 100/200/400/800/1.6k/3.2k/6.4k の順で、一番強い帯域＝0dB。刻みは「表/16分裏/8分裏/16分裏」の立ち上がりの強さ（最大＝1）。", ""]
        rows = []
        for _, r in feat.iterrows():
            if not r["guitar_present"]:
                rows.append([r["section"], CH_JA[r["channel"]], _f(r["level_db"]), "ギターほぼなし", "", "", "", "", "", ""])
                continue
            geq = "/".join(_f(r[f"geq_{l}"], "{:.0f}") for l in GEQ7_LABELS)
            grid = "/".join(_f(r[k], "{:.2f}") for k in ("g1", "g_e", "g_and", "g_a"))
            rows.append([r["section"], CH_JA[r["channel"]], _f(r["level_db"]), _f(r["vs_other_db"], "{:+.1f}"),
                         f"{geq}（最大 {r['geq_peak']}）", _f(r["mono_or_chord"]),
                         f"{_f(r['main_pitch'])}／{_f(r['register_low'])}〜{_f(r['register_high'])}",
                         _f(r["lowest_note"]), _f(r["crest_db"]), grid])
        L += _table(["#", "ch", "音量 dB", "他との差", "GEQ7", "単音/コード", "主な音高／音域", "最低音", "クレスト dB", "刻み"], rows)
        L.append("")

    # ---- ディレイ ----
    dl = _read(R / "delay" / "summary.csv")
    dj = load_json(R / "delay" / "delay.json")
    if dl is not None:
        nl = dj["note_lengths_ms"]
        L += ["## ディレイの判定材料", "",
              f"音符の長さ：4分 {nl['4分']:.0f}ms／付点8分 {nl['付点8分']:.0f}ms／8分 {nl['8分']:.0f}ms。"
              "「過去−未来」の山は ms（一致度の差）。モデルは繰り返しの強さ fb=0／0.3／0.5 で一番一致した遅れ時間。", ""]
        rows = []
        for _, r in dl.iterrows():
            rows.append([r["section"], CH_JA[r["channel"]], r["n_rests"],
                         f"{_f(r['drop80_median'])}（{_f(r['drop80_p25'], '{:.0f}')}〜{_f(r['drop80_p75'], '{:.0f}')}, n={r['drop80_n']}）",
                         _f(r["rest_diff_peaks"]), _f(r["note_diff_peaks"]),
                         " / ".join(_f(r[f"best_D_fb{fb}_ms"], "{:.0f}") for fb in (0.0, 0.3, 0.5)),
                         f"{_f(r['same_at_D'], '{:+.3f}')} / {_f(r['cross_at_D'], '{:+.3f}')}",
                         _f(r["onset_acf_peaks_ref"])])
        L += _table(["#", "ch", "休符", "80ms後の落ち幅 dB", "休符：過去−未来の山", "切り替わり：過去−未来の山",
                     "モデル最大 ms (fb0/0.3/0.5)", "同じch / 反対ch（Dで）", "立ち上がり自己相関（参考）"], rows)
        L += ["", "目安：" + dj["reference"]["drop80"], ""]

    # ---- リバーブ ----
    rv = _read(R / "reverb" / "summary.csv")
    rj = load_json(R / "reverb" / "reverb.json")
    if rv is not None:
        L += ["## リバーブの判定材料", "", "止めた場所の減衰（直前の音＝0dB）。合計チャンネルの値。", ""]
        rows = []
        for _, r in rv[rv.channel == "sum"].iterrows():
            if r["n_stops"] == 0:
                continue
            rows.append([r["section"], r["n_stops"],
                         "/".join(_f(r[k], "{:.0f}") for k in ("v80", "v150", "v250", "v350")),
                         _f(r["slope_100_300"], "{:.0f}"),
                         f"{_f(r['knee_ratio'], '{:.0%}')}（{_f(r['knee_level'], '{:.0f}')}dB, 後 {_f(r['slope_after'], '{:.0f}')}dB/s）",
                         f"{_f(r['tail_LR_corr'], '{:.2f}')} / {_f(r['pre_LR_corr'], '{:.2f}')}",
                         _f(r["tail_chroma_match"], "{:.2f}"),
                         f"{r['n_mix_checked']}（キメ {r['n_all_stop']}）",
                         "/".join(_f(r[k], "{:.0f}") for k in ("mix_v80", "mix_v150", "mix_v250", "mix_v350")),
                         _f(r["side_v80"], "{:.0f}")])
        L += _table(["#", "止め", "80/150/250/350ms dB", "傾き dB/s", "折れ曲がり", "余韻/直前 左右相関",
                     "余韻の音高一致", "原曲で確認", "原曲 80/150/250/350", "原曲 L−R 80ms"], rows)
        sp = rj.get("mix_checked_stops", []) if rj else []
        if sp:
            L += ["", "原曲そのもので確認した止め（他の楽器がほぼ鳴っていない／全員が止まる／曲の最後）："]
            for x in sp[:25]:
                tag = "、".join(t for t, f in (("全員停止", x["all_stop"]), ("他の楽器ほぼなし", x["band_quiet"]),
                                              ("曲の最後", x["is_last"])) if f)
                L.append(f"- {x['stop']}（{tag}）：分離 80ms {_f(x['v80'], '{:.0f}')}dB／原曲 80ms {_f(x['mix_v80'], '{:.0f}')}dB・"
                         f"250ms {_f(x['mix_v250'], '{:.0f}')}dB、折れ曲がり {_f(x['mix_knee_level'], '{:.0f}')}dB・"
                         f"その後 {_f(x['mix_slope_after'], '{:.0f}')}dB/s、L−R 80ms {_f(x['side_v80'], '{:.0f}')}dB")
        L.append("")

    # ---- 揺れ系 ----
    md = _read(R / "modulation" / "summary.csv")
    if md is not None:
        L += ["## 揺れ系の判定材料", "",
              "周期は「秒（拍＝小節、自己相関の強さ）」。谷の範囲は対数周波数の画像で一番深い谷の位置（10〜90%）。", ""]
        rows = []
        for _, r in md.iterrows():
            if "valley_p10_hz" not in r or not np.isfinite(r.get("valley_p10_hz", np.nan)):
                rows.append([r["section"], CH_JA[r["channel"]], "ギターほぼなし", "", "", "", "", "", ""])
                continue
            rows.append([r["section"], CH_JA[r["channel"]],
                         f"{r['valley_p10_hz']:.0f}〜{r['valley_p90_hz']:.0f}Hz",
                         _f(r["valley_periods"]), _f(r["chord_periods"]), _f(r["warning"]),
                         _f(r["notch_count"], "{:.0f}"),
                         f"{_f(r['vib_rate_hz'])}Hz ±{_f(r['vib_depth_cents'], '{:.0f}')}c（{int(r['n_sustained']) if np.isfinite(r['n_sustained']) else 0}音）",
                         f"{_f(r['trem_rate_hz'])}Hz ±{_f(r['trem_depth_db'])}dB"])
        L += _table(["#", "ch", "谷の範囲", "谷の周期の候補", "コードの周期", "注意", "線形の谷の数",
                     "単音の音高の揺れ", "単音の音量の揺れ"], rows)
        L.append("")

    # ---- 踏める場所 ----
    sil = _read(R / "stomp" / "silence.csv")
    sm = _read(R / "stomp" / "stomp.csv")
    if sil is not None or sm is not None:
        L += ["## 踏める場所", ""]
        if sil is not None and len(sil):
            L.append("ギターが完全に無音（左右とも -60dB 以下）の区間：")
            for _, r in sil.iterrows():
                if r["before_guitar"]:
                    continue
                L.append(f"- {r['start_str']}〜{r['end_str']}（{r['length']:.2f}秒）"
                         f"{'　原曲も無音' if r['mix_silent'] else ''}　近い切れ目 #{r['nearest_section']} {r['nearest_boundary']}")
            L.append("")
        if sm is not None and len(sm):
            rows = [[r["section"], r["boundary_str"],
                     f"{r['R_quiet_start']}（{r['R_quiet_db']:.0f}dB）", f"{r['L_quiet_start']}（{r['L_quiet_db']:.0f}dB）",
                     f"{r['LR_quiet_start']}（{r['LR_quiet_db']:.0f}dB）", _f(r["silent_nearby"])]
                    for _, r in sm.iterrows()]
            L += ["切れ目の前後2秒で一番静かな0.15秒："]
            L += _table(["#", "切れ目", "右", "左", "左右とも", "近くの無音"], rows)
        L.append("")

    # ---- 画像 ----
    L += ["## 画像の一覧", ""]
    for sub, desc in (("basic", "テンポと低音の分布"), ("structure", "自己類似・小節ごとの音量"),
                      ("features", "GEQ7・音量と刻み"), ("delay", "パートごとのディレイの曲線・80ms後の落ち幅"),
                      ("reverb", "止めた後の減衰"), ("modulation", "パートごとの山と谷の画像"), ("stomp", "踏める場所")):
        p = R / sub
        if p.exists():
            pngs = sorted(x.name for x in p.glob("*.png"))
            if pngs:
                L.append(f"- {sub}/（{desc}）：" + "、".join(pngs[:6]) + (f" ほか{len(pngs) - 6}枚" if len(pngs) > 6 else ""))
    chk = R / "check"
    if chk.exists():
        for d in sorted(chk.iterdir()):
            if d.is_dir():
                L.append(f"- check/{d.name}/：指定区間の詳細（check.md）")
    L.append("")
    (R / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[summary] {R / 'summary.md'}")
