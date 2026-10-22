"""実行の入口

  python -m analysis.run songs/<曲名>                      全部の解析
  python -m analysis.run songs/<曲名> --only delay         一部だけ再実行（カンマ区切りで複数可）
  python -m analysis.run songs/<曲名> --check 0:25-0:36    指定した時間帯だけ詳しく
  python -m analysis.run songs/<曲名> --reset-sections     sections.csv を自動の区切りで作り直す
  python -m analysis.run songs/<曲名> --downbeat 0:14.45   小節の頭の基準を手で指定（次回以降も使う）
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    # Windows のコンソールで表示できない記号があっても止まらないように
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

from .common import MODULE_ORDER, parse_range  # noqa: E402


def _song_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p
    alt = Path("songs") / arg
    if alt.is_dir():
        return alt
    raise SystemExit(f"曲のフォルダーが見つかりません: {arg}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analysis.run", description="ギターの音作りのための解析")
    ap.add_argument("song", help="曲のフォルダー（例：songs/曲B）")
    ap.add_argument("--only", action="append", default=None,
                    help="一部だけ実行：" + " / ".join(MODULE_ORDER) + "（カンマ区切り可）")
    ap.add_argument("--check", action="append", default=None, help="指定した時間帯を詳しく調べる（例：0:25-0:36）")
    ap.add_argument("--reset-sections", action="store_true", help="sections.csv を自動の区切りで作り直す")
    ap.add_argument("--downbeat", default=None, help="小節の頭にする時間（例：0:14.45）")
    ap.add_argument("--beats-per-bar", type=int, default=None, help="1小節の拍の数（初期値4）")
    args = ap.parse_args(argv)

    song = _song_dir(args.song)
    only = None
    if args.only:
        only = [m.strip() for x in args.only for m in x.split(",") if m.strip()]
        bad = [m for m in only if m not in MODULE_ORDER]
        if bad:
            raise SystemExit(f"--only に使えない名前: {', '.join(bad)}（使えるのは {' / '.join(MODULE_ORDER)}）")
    checks = [parse_range(c) for c in args.check] if args.check else []
    if args.reset_sections or args.downbeat or args.beats_per_bar:
        # 区切りや小節を作り直したら、それを使う解析もやり直す
        if only is not None and "structure" not in only:
            only = ["structure"] + only

    from .context import Context
    from . import basic, structure, features, delay, reverb, modulation, stomp, check, summary
    mods = dict(basic=basic, structure=structure, features=features, delay=delay, reverb=reverb,
                modulation=modulation, stomp=stomp)

    t_start = time.time()
    ctx = Context(song, downbeat=args.downbeat, beats_per_bar=args.beats_per_bar,
                  reset_sections=args.reset_sections)
    if only is not None:
        todo = [m for m in MODULE_ORDER if m in only]
    elif checks:
        todo = []
    else:
        todo = list(MODULE_ORDER)
    failed = []
    for name in todo:
        t0 = time.time()
        try:
            mods[name].run(ctx)
            print(f"  ({name}: {time.time() - t0:.1f}秒)")
        except Exception:
            failed.append(name)
            print(f"[{name}] エラー：", file=sys.stderr)
            traceback.print_exc()
    for a, b in checks:
        try:
            check.run(ctx, a, b)
        except Exception:
            failed.append(f"check {a}-{b}")
            traceback.print_exc()
    try:
        summary.build(ctx)
    except Exception:
        failed.append("summary")
        traceback.print_exc()
    print(f"完了（{time.time() - t_start:.0f}秒）。結果：{ctx.results}")
    if failed:
        print("うまくいかなかったもの：" + ", ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
