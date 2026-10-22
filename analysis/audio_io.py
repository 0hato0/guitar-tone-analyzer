"""io（読み込み）：音源ファイルを探して読み込み、サンプルレートと時間を揃え、「他の楽器」を作る。

- 原曲：ファイル名に「原曲」を含む、または request.md に書いたファイル名
- ギター分離音源：ファイル名に「ギター」「Guitar」を含む、または request.md に書いたファイル名
- 時間の基準は原曲。ギター分離音源のずれ（オフセット）は相互相関で測って補正する
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from scipy import signal

AUDIO_EXT = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


def nfc(s: str) -> str:
    """Mac のファイル名（濁点が分かれた形）を Windows と同じ形に揃える"""
    return unicodedata.normalize("NFC", s)


@dataclass
class SongAudio:
    sr: int
    mix: np.ndarray   # 原曲（2 × サンプル）
    gtr: np.ndarray   # ギター分離音源（2 × サンプル、ずれ補正済み）
    oth: np.ndarray   # 他の楽器＝原曲−ギター
    info: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.mix.shape[1] / self.sr


# ---- ファイルを探す ------------------------------------------------------
def _request_names(song_dir: Path):
    req = song_dir / "request.md"
    if not req.exists():
        return None, None
    text = nfc(req.read_text(encoding="utf-8", errors="replace"))

    def pick(label):
        m = re.search(label + r"[^（(]*[（(]\s*ファイル名\s*[:：]\s*([^）)]*)[）)]", text)
        if m:
            name = m.group(1).strip()
            return name or None
        return None

    return pick("原曲"), pick("ギター分離音源")


def find_inputs(song_dir: Path):
    audio_dir = song_dir / "audio"
    if not audio_dir.exists():
        raise FileNotFoundError(f"audio フォルダーがありません: {audio_dir}")
    files = [p for p in sorted(audio_dir.iterdir())
             if p.is_file() and p.suffix.lower() in AUDIO_EXT and not p.name.startswith(".")]
    if not files:
        raise FileNotFoundError(
            f"音源が見つかりません: {audio_dir}\n"
            "原曲とギター分離音源を入れてください（ファイル名に「原曲」「ギター」を入れる）")
    by_name = {nfc(p.name): p for p in files}
    req_orig, req_gtr = _request_names(song_dir)

    def from_request(name):
        if not name:
            return None
        name = nfc(name)
        if name in by_name:
            return by_name[name]
        # 拡張子なしで書かれた場合
        for n, p in by_name.items():
            if Path(n).stem == name:
                return p
        return None

    gtr = from_request(req_gtr)
    if gtr is None:
        cands = [p for n, p in by_name.items() if "ギター" in n or "guitar" in n.lower()]
        if len(cands) > 1:
            raise RuntimeError("ギター分離音源の候補が複数あります。request.md にファイル名を書いてください: "
                               + ", ".join(p.name for p in cands))
        gtr = cands[0] if cands else None
    orig = from_request(req_orig)
    if orig is None:
        cands = [p for n, p in by_name.items() if "原曲" in n and p != gtr]
        if not cands:
            rest = [p for p in files if p != gtr]
            cands = rest if len(rest) == 1 else []
        if len(cands) > 1:
            raise RuntimeError("原曲の候補が複数あります。request.md にファイル名を書いてください: "
                               + ", ".join(p.name for p in cands))
        orig = cands[0] if cands else None
    if orig is None or gtr is None:
        raise FileNotFoundError(
            "原曲とギター分離音源を見分けられませんでした。ファイル名に「原曲」「ギター」を入れるか、"
            f"request.md にファイル名を書いてください。audio の中身: {', '.join(p.name for p in files)}")
    return orig, gtr


# ---- 読み込み ------------------------------------------------------------
def read_audio(path: Path):
    """2 × サンプル の float32 と サンプルレート。日本語のパスでも読めるようにファイルを開いて渡す"""
    try:
        with open(path, "rb") as f:
            data, sr = sf.read(f, dtype="float32", always_2d=True)
        return data.T.copy(), int(sr)
    except Exception as e_sf:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                f"{path.name} を読めませんでした（{e_sf}）。wav か mp3 で保存し直すか、ffmpeg を入れてください") from e_sf
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "tmp.wav"
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(path), str(out)], check=True)
            with open(out, "rb") as f:
                data, sr = sf.read(f, dtype="float32", always_2d=True)
        return data.T.copy(), int(sr)


def _stereo(y: np.ndarray) -> np.ndarray:
    if y.shape[0] == 1:
        return np.vstack([y, y])
    return y[:2]


def _shift(y: np.ndarray, lag: int) -> np.ndarray:
    """lag > 0 なら後ろへずらす（先頭に無音を足す）"""
    if lag == 0:
        return y
    out = np.zeros_like(y)
    if lag > 0:
        out[:, lag:] = y[:, :-lag]
    else:
        out[:, :lag] = y[:, -lag:]
    return out


def estimate_offset(mix: np.ndarray, gtr: np.ndarray, sr: int, max_lag_s: float = 5.0):
    """原曲の中のギターが、ギター分離音源より何サンプル後ろにあるか（相互相関）"""
    low = 8000
    x = librosa.resample(mix.mean(axis=0), orig_sr=sr, target_sr=low)
    g = librosa.resample(gtr.mean(axis=0), orig_sr=sr, target_sr=low)
    n = min(len(x), len(g))
    x, g = x[:n], g[:n]
    r = signal.correlate(x, g, mode="full", method="fft")
    lags = signal.correlation_lags(len(x), len(g), mode="full")
    sel = np.abs(lags) <= int(max_lag_s * low)
    r, lags = r[sel], lags[sel]
    i = int(np.argmax(np.abs(r)))
    coarse = lags[i] * sr / low
    peak_ratio = float(np.abs(r[i]) / (np.median(np.abs(r)) + 1e-12))

    # 元のサンプルレートで細かく合わせる（ギターが大きい30秒を使う）
    xs, gs = mix.mean(axis=0), gtr.mean(axis=0)
    win = int(30 * sr)
    if len(gs) > win:
        e = np.convolve(gs[::sr // 10] ** 2, np.ones(300), mode="valid")
        start = int(np.argmax(e)) * (sr // 10)
        start = min(max(start, int(sr * 6)), len(gs) - win - int(sr * 6))
        start = max(start, 0)
    else:
        start, win = 0, len(gs)
    best, best_lag = -np.inf, int(round(coarse))
    span = int(np.ceil(sr / low)) + 2
    seg_g = gs[start:start + win]
    for lag in range(int(round(coarse)) - span, int(round(coarse)) + span + 1):
        a = start + lag
        if a < 0 or a + win > len(xs):
            continue
        v = abs(float(np.dot(xs[a:a + win], seg_g)))
        if v > best:
            best, best_lag = v, lag
    sign = 1.0 if r[i] >= 0 else -1.0
    return best_lag, sign, peak_ratio


def load_song(song_dir: Path) -> SongAudio:
    orig_p, gtr_p = find_inputs(song_dir)
    mix, sr_m = read_audio(orig_p)
    gtr, sr_g = read_audio(gtr_p)
    info = {
        "files": {"原曲": orig_p.name, "ギター": gtr_p.name},
        "original": {
            "原曲": {"sr": sr_m, "channels": int(mix.shape[0]), "duration_s": mix.shape[1] / sr_m},
            "ギター": {"sr": sr_g, "channels": int(gtr.shape[0]), "duration_s": gtr.shape[1] / sr_g},
        },
    }
    mix, gtr = _stereo(mix), _stereo(gtr)
    sr = min(sr_m, sr_g)   # 低い方に揃える（高い方を下げる）
    if sr_m != sr:
        mix = librosa.resample(mix, orig_sr=sr_m, target_sr=sr, axis=-1).astype(np.float32)
    if sr_g != sr:
        gtr = librosa.resample(gtr, orig_sr=sr_g, target_sr=sr, axis=-1).astype(np.float32)
    n = min(mix.shape[1], gtr.shape[1])
    mix, gtr = mix[:, :n], gtr[:, :n]

    lag, sign, peak_ratio = estimate_offset(mix, gtr, sr)
    gtr = _shift(gtr, lag)
    if sign < 0:
        gtr = -gtr
    # 他の楽器＝原曲−ギター。分離音源の音量が原曲とずれている場合だけ倍率を合わせる
    gains = []
    for c in range(2):
        den = float(np.dot(gtr[c], gtr[c]))
        gains.append(float(np.dot(mix[c], gtr[c]) / den) if den > 0 else 1.0)
    gain = float(np.mean(gains))
    use_gain = 1.0 if 0.7 <= gain <= 1.4 else gain
    oth = (mix - use_gain * gtr).astype(np.float32)

    info.update({
        "analysis_sr": sr,
        "duration_s": n / sr,
        "offset": {
            "samples": int(lag), "ms": 1000.0 * lag / sr,
            "meaning": "正の値＝ギター分離音源を後ろにずらして原曲に合わせた",
            "polarity_inverted": bool(sign < 0),
            "peak_ratio": peak_ratio,
            "corrected": bool(lag != 0 or sign < 0),
        },
        "guitar_gain_ls": gain,
        "guitar_gain_used": use_gain,
        "channels_note": "モノラルの音源は左右に同じ音を入れて扱う",
    })
    return SongAudio(sr=sr, mix=mix.astype(np.float32), gtr=gtr.astype(np.float32), oth=oth, info=info)
