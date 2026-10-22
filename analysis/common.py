"""共通の小道具：定数、時間の表示と読み取り、ファイル保存。"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---- 解析の基本設定 -------------------------------------------------------
HOP = 512            # CQT・クロマ・立ち上がり強度のホップ（44.1kHzで約11.6ms）
ENV_FRAME = 1024     # 音量カーブの窓（約23ms）
ENV_HOP = 256        # 音量カーブのホップ（約5.8ms）
EPS = 1e-10

# CQT：C2（65.4Hz）から6オクターブ、半音ごと
CQT_FMIN_MIDI = 36   # C2
CQT_BINS = 72

# 鳴っている判定：その音源の大きい所（99パーセンタイル）から何dB下までを「鳴っている」とするか
ACTIVE_RANGE_DB = 35.0

GEQ7_CENTERS = [100, 200, 400, 800, 1600, 3200, 6400]
GEQ7_LABELS = ["100", "200", "400", "800", "1.6k", "3.2k", "6.4k"]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHANNELS = ("sum", "L", "R")
CH_JA = {"sum": "合計", "L": "左", "R": "右"}
KIND_JA = {"mix": "原曲", "gtr": "ギター", "oth": "他の楽器"}

MODULE_ORDER = ["basic", "structure", "features", "delay", "reverb", "modulation", "stomp"]


# ---- 数値 ----------------------------------------------------------------
def db(x):
    """振幅 → dB"""
    return 20.0 * np.log10(np.maximum(np.abs(x), EPS))


def pow_db(p):
    """パワー → dB"""
    return 10.0 * np.log10(np.maximum(p, EPS))


def rms_db(y) -> float:
    if y is None or len(y) == 0:
        return float("nan")
    return float(pow_db(np.mean(np.square(y, dtype=np.float64))))


def corr(a, b) -> float:
    """ピアソン相関（どちらかが無音なら NaN）"""
    if len(a) < 2:
        return float("nan")
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if den <= 0:
        return float("nan")
    return float(np.dot(a, b) / den)


def cos_sim(a, b) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def midi_name(m) -> str:
    if m is None or not np.isfinite(m):
        return ""
    m = int(round(float(m)))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def nan_median(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


# ---- 時間 ----------------------------------------------------------------
def fmt_time(sec, digits: int = 2) -> str:
    """秒 → 「分:秒」（例：74.3 → 1:14.30）"""
    if sec is None:
        return ""
    try:
        sec = float(sec)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(sec):
        return ""
    sign = "-" if sec < 0 else ""
    sec = abs(sec)
    m = int(sec // 60)
    s = round(sec - 60 * m, digits)
    if s >= 60:
        m += 1
        s -= 60
    width = 3 + digits if digits > 0 else 2
    return f"{sign}{m}:{s:0{width}.{digits}f}"


def parse_time(text) -> float:
    """「1:14.3」「74.3」「1分14.3秒」→ 秒"""
    if isinstance(text, (int, float, np.integer, np.floating)):
        return float(text)
    t = str(text).strip().replace("：", ":").replace("分", ":").replace("秒", "")
    if t == "":
        raise ValueError("時間が空です")
    parts = t.split(":")
    val = 0.0
    for p in parts:
        val = val * 60 + float(p)
    return val


def parse_range(text: str):
    """「0:25-0:36」「0:25〜0:36」→ (25.0, 36.0)"""
    t = str(text).strip()
    m = re.split(r"\s*(?:-|〜|~|～|–|—)\s*", t, maxsplit=1)
    if len(m) != 2:
        raise ValueError(f"時間の範囲が読めません: {text}（例：0:25-0:36）")
    a, b = parse_time(m[0]), parse_time(m[1])
    if b <= a:
        raise ValueError(f"終わりが始まりより前です: {text}")
    return a, b


def safe_name(text: str) -> str:
    """ファイル名に使えない文字を置き換える"""
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(text)).strip("_") or "x"


# ---- 保存 ----------------------------------------------------------------
def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else round(f, 6)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, ensure_ascii=False, indent=2)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: Path, decimals: int = 4) -> None:
    """Excel（Windows）でも日本語が化けないよう BOM 付き UTF-8 で保存"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(decimals)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")
