# analysis（解析スクリプト）

原曲とギター分離音源を解析し、音作りの判断材料（数値・画像）を `songs/<曲名>/results/` に残すツールです。
**「ディレイあり」などの判定はしません。** 判定は Claude Code が数値と画像を見て、CLAUDE.md のルールに沿って行います。しきい値は参考値として出すだけです。

## 実行方法

プロジェクトの一番上のフォルダーで実行します（Windows は `.venv\Scripts\python`、Mac は `.venv/bin/python`）。

```
python -m analysis.run songs/<曲名>                      全部の解析
python -m analysis.run songs/<曲名> --only delay         一部だけ再実行（カンマ区切り可）
                                                          basic / structure / features / delay / reverb / modulation / stomp
python -m analysis.run songs/<曲名> --check 0:25-0:36    指定した時間帯だけ詳しく（複数指定可）
python -m analysis.run songs/<曲名> --reset-sections     sections.csv を自動の区切りで作り直す
python -m analysis.run songs/<曲名> --downbeat 0:14.45   小節の頭の基準を手で指定（次回以降も使う）
python -m analysis.run songs/<曲名> --beats-per-bar 3    1小節の拍の数（初期値4）
```

最初の準備：`python -m venv .venv` → `.venv\Scripts\python -m pip install -r requirements.txt`

## モジュールの役割

| ファイル | 役割 |
|---|---|
| `audio_io.py` | 音源を探して読み込む（wav・mp3、ffmpeg があれば m4a なども）。サンプルレートを低い方に揃え、長さを短い方に揃え、ギター分離音源の時間のずれを相互相関で補正する。「他の楽器」＝原曲−ギター を作る |
| `context.py` | 読み込んだ音源、拍・小節、構成の区切り、計算済みの特徴量を各モジュールで共有する |
| `basic.py` | 長さ・サンプルレート・チャンネル数、テンポ（全体と20秒ごと）、チューニング、最低音 |
| `structure.py` | 拍の追跡、ギターの最初の音を基準にした小節、小節ごとの表、構成の切れ目の候補（自己類似） |
| `features.py` | パートごと・合計／左／右ごとの音量、他の楽器との差、Gt GEQ7 のバランス、左右の差と相関、音高と音域、単音かコードか、クレストファクター、16分のグリッドごとの立ち上がり |
| `delay.py` | 短い休符の80ms後の落ち幅、休符の音高と「過去／未来」の音の一致度、音の切り替わりでの残り、フィードバック込みのモデルの当てはめ、左右どちらに繰り返しが出るか |
| `reverb.py` | 止めた場所の減衰カーブ（20〜600ms）、傾き、折れ曲がり、余韻の左右相関と音高。他の楽器が鳴っていない場面と全員停止は原曲でも確認 |
| `modulation.py` | 倍音をならしたスペクトログラム（対数・線形）の山と谷の画像、谷の動きの周期とコードの周期、伸ばした単音の音高・音量の揺れ |
| `stomp.py` | 切れ目の前後で一番静かな0.15秒、ギターが -60dB 以下の無音区間 |
| `check.py` | `--check` の時間帯の詳細（左右の音高、山と谷、休符の残り方、ディレイの曲線） |
| `summary.py` | 主な数値を `results/summary.md` にまとめる |
| `dsp.py`・`plotting.py`・`common.py` | 共通の信号処理、図（横軸「分:秒」、拍・小節・構成の線）、保存 |

## 出力ファイル（`songs/<曲名>/results/`）

| ファイル | 見方 |
|---|---|
| `summary.md` | 主な数値の表。まずここを見る |
| `sections.csv` | 構成の区切り（開始・終了・パート名）。**手で直すと次回からそれが使われる** |
| `settings.json` | `--downbeat` などの手動指定 |
| `basic/` | `basic.json`、`tempo_windows.csv`（20秒ごとのテンポ。拍の取り違えの注意付き）、`basic.png` |
| `structure/` | `beats.csv`、`bars.csv`（小節ごとの音量・左右・クロマ）、`sections_auto.csv`（自動の区切り候補）、`ssm.png`、`bars.png` |
| `features/` | `features.csv`、`geq7.png`（パート×帯域）、`levels_grid.png` |
| `delay/` | `summary.csv`、`rests.csv`（休符ごと）、`curves.csv`、パートごとの曲線の図、`drop80.png` |
| `reverb/` | `summary.csv`、`stops.csv`、`curves.csv`、`decay_*.png`、`mix_decay.png` |
| `modulation/` | `summary.csv`、`sustained_notes.csv`、パートごとの山と谷の図（赤＝いつもより強い、青＝谷） |
| `stomp/` | `stomp.csv`、`silence.csv`、`stomp.png` |
| `check/<範囲>/` | `check.md`、`pitch_LR.png`、`modulation_LR.png`、`rests.png`、`delay.png` |

## 既知の弱点（レポートに使うときの注意）

- **最低音**：CQT の一番低いビン（C2）がいつも「一番低い音」と出る不具合がある。最低音の数値は使わない
- **テンポ**：合成の試験曲（175BPM）で半分（87.6BPM）と出た。`tempo_windows.csv` と `basic.png` で、半分・倍になっていないか必ず確認する
- **ディレイの遅れ時間**：休符の音高を過去の窓と比べる方法なので、音符1つ分の幅でずれる（試験曲：正解 290ms → 推定 310〜350ms）。遅れ時間は画像と合わせて判断し、特定できなければ試す順番を示す
- **完全な無音（デジタル0）**：落ち幅が -180dB のような意味のない値になる
- **揺れの周期**：コード進行の周期と重なると区別できない（試験曲でフェイザー 2.74秒 に対し、コードの周期 5.48秒 が出た）。注意の列に出る
- **モノラル音源**：左右の差・相関・L−R を使う判定は意味がない。`summary.md` の左右相関がどこも 1.00 ならモノラル
- **分離音源の質**：キーボードなどが常に混ざると、休符の落ち幅がどのパートでも同じくらいになり、パートごとの特徴として使えない

## 変更履歴

- 2026-10-22：初版（初回セットアップ）。合成の試験曲（ディレイ・フェイザー・キメ入り）で全モジュールの動作を確認
- 2026-10-22：パート名に「?」などファイル名に使えない文字があると、delay・modulation の図の保存で止まる問題を修正（`context.py` でファイル名用に置き換え）
