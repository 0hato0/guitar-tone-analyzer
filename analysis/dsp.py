"""信号処理の部品（複数のモジュールで共通に使うもの）。"""
from __future__ import annotations

import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import maximum_filter1d

from .common import (CQT_BINS, CQT_FMIN_MIDI, ENV_FRAME, ENV_HOP, EPS, HOP,
                     db, pow_db)

# 2〜8倍音が基音から何半音上にあるか（音の数を数えるときに倍音を別の音と数えないため）
HARMONIC_OFFSETS = (12, 19, 24, 28, 31, 34, 36)


# ---- 音量 ----------------------------------------------------------------
def level_env(y: np.ndarray) -> np.ndarray:
    """音量カーブ（dBFS、窓 ENV_FRAME、ホップ ENV_HOP）"""
    r = librosa.feature.rms(y=y, frame_length=ENV_FRAME, hop_length=ENV_HOP, center=True)[0]
    return db(r).astype(np.float32)


def moving_level_db(y: np.ndarray, sr: int, win_s: float, hop: int = ENV_HOP):
    """長さ win_s の窓の平均パワー（dB）を hop ごとに。返り値：(開始時刻, dB)"""
    p = np.concatenate([[0.0], np.cumsum(np.square(y, dtype=np.float64))])
    n = max(1, int(round(win_s * sr)))
    starts = np.arange(0, max(1, len(y) - n + 1), hop)
    vals = (p[starts + n] - p[starts]) / n
    return starts / sr, pow_db(vals)


def past_max(x: np.ndarray, n: int) -> np.ndarray:
    """x[i-n .. i] の最大値（直前の音の大きさ）"""
    if n <= 0:
        return x.copy()
    # maximum_filter1d の窓を「過去側」だけにずらす
    return maximum_filter1d(x, size=n + 1, origin=n // 2, mode="nearest")


# ---- 音高 ----------------------------------------------------------------
def cqt_mag(y: np.ndarray, sr: int, tuning: float = 0.0) -> np.ndarray:
    """半音ごとのCQT振幅（C2〜B7、72本 × フレーム）"""
    C = librosa.cqt(y, sr=sr, hop_length=HOP, fmin=librosa.midi_to_hz(CQT_FMIN_MIDI),
                    n_bins=CQT_BINS, bins_per_octave=12, tuning=tuning)
    return np.abs(C).astype(np.float32)


def chroma_from_cqt(C: np.ndarray) -> np.ndarray:
    """CQT → クロマ（12 × フレーム、C始まり）"""
    out = np.zeros((12, C.shape[1]), dtype=np.float32)
    for k in range(C.shape[0]):
        out[(CQT_FMIN_MIDI + k) % 12] += C[k]
    return out


def frame_notes(col: np.ndarray, rel_db: float = -15.0, max_notes: int = 6):
    """1フレームのCQTから、倍音を除いた「鳴っている音」（ビン番号）を低い順に推定する"""
    m = float(col.max())
    if m <= 0:
        return []
    thr = m * 10 ** (rel_db / 20)
    n = len(col)
    cand = [k for k in range(n)
            if col[k] >= thr and (k == 0 or col[k] >= col[k - 1]) and (k == n - 1 or col[k] >= col[k + 1])]
    notes, blocked = [], set()
    for k in cand:
        if k in blocked:
            continue
        notes.append(k)
        for h in HARMONIC_OFFSETS:
            blocked.add(k + h)
        if len(notes) >= max_notes:
            break
    return notes


def note_tracks(C: np.ndarray, active: np.ndarray):
    """フレームごとの「音の数」と「一番低い音（MIDI）」。鳴っていないフレームは NaN"""
    n = C.shape[1]
    count = np.full(n, np.nan, dtype=np.float32)
    lowest = np.full(n, np.nan, dtype=np.float32)
    for i in np.flatnonzero(active[:n]):
        notes = frame_notes(C[:, i])
        if notes:
            count[i] = len(notes)
            lowest[i] = CQT_FMIN_MIDI + notes[0]
    return count, lowest


# ---- 窓の平均プロファイル（ディレイの判定用） ------------------------------
class ProfileBank:
    """CQT の任意区間の平均を高速に取り出す（累積和）"""

    def __init__(self, C: np.ndarray):
        self.n = C.shape[1]
        self.cs = np.concatenate([np.zeros((C.shape[0], 1), np.float64),
                                  np.cumsum(C, axis=1, dtype=np.float64)], axis=1)

    def get(self, a: int, b: int):
        """フレーム a 以上 b 未満の平均。範囲外なら None"""
        if a < 0 or b > self.n or b <= a:
            return None
        return (self.cs[:, b] - self.cs[:, a]) / (b - a)


# ---- 休符・止めの検出 ---------------------------------------------------
def find_rests(env, hop_s, i0, i1, drop_db=10.0, min_len=0.06, ref_win=0.15, min_ref=-np.inf):
    """短い休符：直前の音（過去 ref_win 秒の最大）より drop_db 以上下がり、min_len 秒以上続く場所。
    返り値：dict のリスト（フレーム番号）。stop＝音を止めた瞬間（直前の音から3dB下がる前の最後のフレーム）"""
    n_ref = max(1, int(round(ref_win / hop_s)))
    ref = past_max(env, n_ref)
    out = []
    i = max(i0, n_ref)
    i1 = min(i1, len(env))
    while i < i1:
        r = ref[i]
        if r > min_ref and env[i] < r - drop_db:
            j = i
            while j < i1 and env[j] < r - drop_db:
                j += 1
            k = i
            while k > i - n_ref and env[k] < r - 3:
                k -= 1
            if (j - k) * hop_s >= min_len and j < i1:   # 曲や区間の端で切れたものは除く
                out.append(dict(stop=k, start=i, end=j, ref=float(r)))
            i = j
        else:
            i += 1
    return out


def find_stops(env, hop_s, i0, i1, drop_db=20.0, within=0.15, back_db=10.0, quiet=0.4,
               min_ref=-np.inf):
    """ギターを止めた場所：within 秒以内に drop_db 以上落ち、quiet 秒以上 back_db まで戻らない場所"""
    n_ref = max(1, int(round(within / hop_s)))
    ref = past_max(env, n_ref)
    n_quiet = int(round(quiet / hop_s))
    out = []
    i = max(i0, n_ref)
    i1 = min(i1, len(env))
    while i < i1:
        r = ref[i]
        if r > min_ref and env[i] < r - drop_db:
            k = i
            while k > i - n_ref and env[k] < r - 3:
                k -= 1
            j = i
            while j < len(env) and env[j] < r - back_db:
                j += 1
            if j - i >= n_quiet:
                out.append(dict(stop=k, start=i, back=j, ref=float(r)))
            i = max(j, i + 1)
        else:
            i += 1
    return out


def knee_fit(t, v):
    """減衰カーブ（秒, dB）を2本の直線で近似し、折れ曲がりを探す"""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    res = dict(knee_time=np.nan, knee_level=np.nan, slope_before=np.nan,
               slope_after=np.nan, is_knee=False)
    if len(t) < 8:
        return res
    p1 = np.polyfit(t, v, 1)
    sse1 = float(np.sum((np.polyval(p1, t) - v) ** 2))
    best = None
    for k in range(3, len(t) - 3):
        a = np.polyfit(t[:k + 1], v[:k + 1], 1)
        b = np.polyfit(t[k:], v[k:], 1)
        sse = float(np.sum((np.polyval(a, t[:k + 1]) - v[:k + 1]) ** 2)
                    + np.sum((np.polyval(b, t[k:]) - v[k:]) ** 2))
        if best is None or sse < best[0]:
            best = (sse, k, a, b)
    if best is None:
        return res
    sse, k, a, b = best
    res.update(knee_time=float(t[k]), knee_level=float(np.polyval(b, t[k])),
               slope_before=float(a[0]), slope_after=float(b[0]))
    # 「急に落ちてから緩やかに減る」：前半が後半より2倍以上急で、2本にすると誤差が4割以上減る
    res["is_knee"] = bool(a[0] < 0 and a[0] < 2 * min(b[0], -1e-6) and sse < 0.6 * sse1)
    return res


# ---- 周期 ----------------------------------------------------------------
def acf(x: np.ndarray) -> np.ndarray:
    """正規化した自己相関（ラグ0〜）。NaN は平均で埋める"""
    x = np.asarray(x, float).copy()
    m = np.nanmean(x) if np.any(np.isfinite(x)) else 0.0
    x[~np.isfinite(x)] = m
    x = x - x.mean()
    if np.allclose(x, 0):
        return np.zeros(len(x))
    r = signal.correlate(x, x, mode="full", method="fft")[len(x) - 1:]
    # 重なりの長さで割ってから正規化（長いラグで小さくなりすぎないように）
    r = r / np.arange(len(x), 0, -1)
    return r / r[0]


def top_peaks(x_axis, y, n=3, prominence=None, min_x=None, max_x=None):
    """曲線の山を高い順に n 個（x, y）"""
    x_axis = np.asarray(x_axis, float)
    y = np.asarray(y, float)
    sel = np.isfinite(y)
    if min_x is not None:
        sel &= x_axis >= min_x
    if max_x is not None:
        sel &= x_axis <= max_x
    if sel.sum() < 3:
        return []
    xs, ys = x_axis[sel], y[sel]
    if prominence is None:
        prominence = 0.1 * (np.nanmax(ys) - np.nanmin(ys) + EPS)
    idx, _ = signal.find_peaks(ys, prominence=prominence)
    idx = sorted(idx, key=lambda i: -ys[i])[:n]
    return [(float(xs[i]), float(ys[i])) for i in idx]


def bandpass(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    sos = signal.butter(4, [lo, min(hi, sr / 2 * 0.95)], btype="band", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y).astype(np.float32)


def stft_power(y: np.ndarray, n_fft: int = 4096, hop: int = 1024) -> np.ndarray:
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True)
    return (np.abs(S) ** 2).astype(np.float32)


def band_powers(P_mean: np.ndarray, freqs: np.ndarray, centers) -> np.ndarray:
    """1オクターブ幅の帯域ごとのパワー合計（dB）"""
    out = []
    for fc in centers:
        sel = (freqs >= fc / np.sqrt(2)) & (freqs < fc * np.sqrt(2))
        out.append(float(pow_db(P_mean[sel].sum())))
    return np.array(out)
