"""図の共通処理：日本語フォント、横軸「分:秒」、拍・小節・構成の線。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

_JP_FONTS = ["Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "Hiragino Kaku Gothic ProN",
             "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic", "TakaoGothic"]


def setup_fonts():
    have = {f.name for f in font_manager.fontManager.ttflist}
    fams = [f for f in _JP_FONTS if f in have]
    plt.rcParams["font.family"] = fams + ["DejaVu Sans"] if fams else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 110


setup_fonts()


def _mmss(x, _pos=None):
    sign = "-" if x < 0 else ""
    x = abs(x)
    m = int(x // 60)
    s = x - 60 * m
    if abs(s - round(s)) < 1e-6:
        return f"{sign}{m}:{int(round(s)) % 60:02d}" if round(s) < 60 else f"{sign}{m + 1}:00"
    return f"{sign}{m}:{s:04.1f}"


def time_axis(ax, nbins: int = 12):
    ax.xaxis.set_major_formatter(FuncFormatter(_mmss))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]))


def draw_grid(ax, ctx, t0: float, t1: float, sections: bool = True, labels: bool = True):
    """拍（細い線）・小節（灰色の線）・構成の切れ目（赤い破線）"""
    span = t1 - t0
    bi = ctx.beat_info()
    beats, bib, bars = bi["beats"], bi["beat_in_bar"], bi["bar"]
    sel = (beats >= t0) & (beats <= t1)
    if span <= 45:
        for b in beats[sel & (bib != 1)]:
            ax.axvline(b, color="0.75", lw=0.4, alpha=0.6, zorder=1)
    if span <= 200:
        for b, n in zip(beats[sel & (bib == 1)], bars[sel & (bib == 1)]):
            ax.axvline(b, color="0.35", lw=0.7, alpha=0.6, zorder=1)
            if labels and span <= 45:
                ax.text(b, 1.0, f"{n}", transform=ax.get_xaxis_transform(), fontsize=6,
                        color="0.3", va="bottom", ha="left")
    if sections:
        for s in ctx.sections():
            if t0 <= s["start"] <= t1:
                ax.axvline(s["start"], color="crimson", lw=1.0, ls="--", alpha=0.8, zorder=2)
                if labels:
                    ax.text(s["start"], 1.0, f" {s['idx']:02d}{(' ' + s['name']) if s['name'] else ''}",
                            transform=ax.get_xaxis_transform(), fontsize=7, color="crimson",
                            va="bottom", ha="left")
    ax.set_xlim(t0, t1)
    time_axis(ax)


def note_value_lines(ax, ctx, vertical: bool = True, ymin=None):
    """4分・付点8分・8分の長さ（秒）を縦線で"""
    q = ctx.beat_s
    marks = [(q, "4分", "tab:blue"), (0.75 * q, "付点8分", "tab:orange"), (0.5 * q, "8分", "tab:green"),
             (1.5 * q, "付点4分", "tab:purple"), (2 * q, "2分", "tab:brown")]
    for v, name, c in marks:
        if ax.get_xlim()[0] <= v <= ax.get_xlim()[1]:
            ax.axvline(v, color=c, lw=1, ls=":", alpha=0.9)
            ax.text(v, 1.0, f"{name}\n{v * 1000:.0f}ms", transform=ax.get_xaxis_transform(),
                    fontsize=6, color=c, ha="center", va="bottom")


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def note_axis(ax, midi_lo: int, midi_hi: int, origin_midi: int):
    """縦軸をMIDIビン番号 → 音名（Eごと）に"""
    from .common import midi_name
    ticks = [m for m in range(midi_lo, midi_hi + 1) if m % 12 in (4,)]  # E
    ax.set_yticks([m - origin_midi for m in ticks])
    ax.set_yticklabels([midi_name(m) for m in ticks])


def image_extent(t0, t1, y0, y1):
    return [t0, t1, y0, y1]


def robust_clim(Z, pct=98):
    v = np.nanpercentile(np.abs(Z), pct) if np.any(np.isfinite(Z)) else 1.0
    return max(v, 1e-3)
