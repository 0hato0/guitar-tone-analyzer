"""解析中に共有するデータ（読み込んだ音源、拍・小節、構成の区切り、計算済みの特徴量）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from . import audio_io, dsp
from .common import (ACTIVE_RANGE_DB, ENV_HOP, HOP, fmt_time, load_json,
                     parse_time, read_csv, safe_name, save_json)


class Context:
    def __init__(self, song_dir: Path, downbeat=None, beats_per_bar=None, reset_sections=False):
        self.song_dir = Path(song_dir).resolve()
        self.results = self.song_dir / "results"
        self.results.mkdir(parents=True, exist_ok=True)
        self.song_name = self.song_dir.name

        # 拍子・小節の頭の手動指定は results/settings.json に残して次回も使う
        self.settings_path = self.results / "settings.json"
        self.settings = load_json(self.settings_path, {}) or {}
        changed = False
        if downbeat is not None:
            self.settings["downbeat"] = fmt_time(parse_time(downbeat), 3)
            changed = True
        if beats_per_bar is not None:
            self.settings["beats_per_bar"] = int(beats_per_bar)
            changed = True
        if changed:
            save_json(self.settings_path, self.settings)
        self.reset_sections = reset_sections

        print(f"[io] 音源を読み込み中: {self.song_dir}")
        self.audio = audio_io.load_song(self.song_dir)
        self.sr = self.audio.sr
        self.duration = self.audio.duration
        self._cache = {}
        self._beat_info = None
        self._beats_fresh = False
        self._sections = None
        off = self.audio.info["offset"]
        print(f"[io] {self.audio.info['files']['原曲']} / {self.audio.info['files']['ギター']}"
              f"  解析SR={self.sr}Hz  長さ={fmt_time(self.duration)}  ずれ補正={off['ms']:.1f}ms")

    # ---- 出力先 ----------------------------------------------------------
    def subdir(self, name: str) -> Path:
        p = self.results / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def beats_per_bar(self) -> int:
        return int(self.settings.get("beats_per_bar", 4))

    # ---- 信号 ------------------------------------------------------------
    def sig(self, kind: str, ch: str) -> np.ndarray:
        """kind: mix / gtr / oth、ch: sum / L / R / side（L−R）"""
        y = getattr(self.audio, kind)
        if ch == "L":
            return y[0]
        if ch == "R":
            return y[1]
        key = ("sig", kind, ch)
        if key not in self._cache:
            if ch == "sum":
                self._cache[key] = (0.5 * (y[0] + y[1])).astype(np.float32)
            elif ch == "side":
                self._cache[key] = (0.5 * (y[0] - y[1])).astype(np.float32)
            else:
                raise ValueError(ch)
        return self._cache[key]

    def _cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # 音量カーブ（ENV_HOP ごと）
    def env(self, kind, ch):
        return self._cached(("env", kind, ch), lambda: dsp.level_env(self.sig(kind, ch)))

    @property
    def env_hop_s(self) -> float:
        return ENV_HOP / self.sr

    def env_index(self, t: float) -> int:
        return int(round(t / self.env_hop_s))

    def ref_level(self, kind) -> float:
        """その音源の「大きい所」の音量（合計の99パーセンタイル）"""
        return self._cached(("ref", kind), lambda: float(np.percentile(self.env(kind, "sum"), 99)))

    def active_threshold(self, kind) -> float:
        return self.ref_level(kind) - ACTIVE_RANGE_DB

    # 特徴量のフレーム（HOP ごと）
    @property
    def hop_s(self) -> float:
        return HOP / self.sr

    def frame_index(self, t: float) -> int:
        return int(round(t / self.hop_s))

    def tuning(self) -> float:
        """A=440Hz からのずれ（半音の何分の1か、-0.5〜0.5）"""
        def calc():
            y = self.sig("gtr", "sum")
            return float(librosa.estimate_tuning(y=y, sr=self.sr, n_fft=4096, resolution=0.01))
        return self._cached(("tuning",), calc)

    def cqt(self, kind, ch):
        return self._cached(("cqt", kind, ch), lambda: dsp.cqt_mag(self.sig(kind, ch), self.sr, self.tuning()))

    def chroma(self, kind, ch):
        return self._cached(("chroma", kind, ch), lambda: dsp.chroma_from_cqt(self.cqt(kind, ch)))

    def active_frames(self, kind, ch) -> np.ndarray:
        """HOP ごとのフレームが鳴っているか"""
        def calc():
            env = self.env(kind, ch)
            n = self.cqt(kind, ch).shape[1]
            idx = np.clip(np.round(np.arange(n) * HOP / ENV_HOP).astype(int), 0, len(env) - 1)
            return env[idx] > self.active_threshold(kind)
        return self._cached(("active", kind, ch), calc)

    def notes(self, kind, ch):
        """(音の数, 一番低い音のMIDI) をフレームごとに"""
        return self._cached(("notes", kind, ch),
                            lambda: dsp.note_tracks(self.cqt(kind, ch), self.active_frames(kind, ch)))

    def onset_env(self, kind, ch):
        return self._cached(("onset", kind, ch),
                            lambda: librosa.onset.onset_strength(y=self.sig(kind, ch), sr=self.sr, hop_length=HOP))

    def onsets(self, kind, ch):
        """立ち上がりのフレーム番号"""
        return self._cached(("onsets", kind, ch), lambda: librosa.onset.onset_detect(
            onset_envelope=self.onset_env(kind, ch), sr=self.sr, hop_length=HOP, units="frames"))

    # ---- 拍・小節 --------------------------------------------------------
    def beat_info(self, force: bool = False) -> dict:
        """拍（ビート追跡）と小節の頭。structure を実行した回はその場で計算し直す"""
        if self._beat_info is not None and (self._beats_fresh or not force):
            return self._beat_info
        path = self.results / "structure" / "beats.csv"
        meta_path = self.results / "structure" / "beat_meta.json"
        if not force and path.exists() and meta_path.exists():
            df = read_csv(path)
            meta = load_json(meta_path)
            if abs(meta.get("duration_s", 0) - self.duration) < 0.5:
                self._beat_info = dict(meta, beats=df["time"].to_numpy(float),
                                       beat_in_bar=df["beat_in_bar"].to_numpy(int),
                                       bar=df["bar"].to_numpy(int),
                                       extrapolated=df["extrapolated"].to_numpy(bool))
                return self._beat_info
        from . import structure
        self._beat_info = structure.compute_beats(self)
        self._beats_fresh = True
        return self._beat_info

    @property
    def tempo(self) -> float:
        return float(self.beat_info()["tempo"])

    @property
    def beat_s(self) -> float:
        return 60.0 / self.tempo

    def beats(self) -> np.ndarray:
        return self.beat_info()["beats"]

    def downbeats(self) -> np.ndarray:
        bi = self.beat_info()
        return bi["beats"][bi["beat_in_bar"] == 1]

    # ---- 構成 ------------------------------------------------------------
    def sections(self) -> list[dict]:
        """results/sections.csv（手で直したものを優先）。なければ structure を実行して作る"""
        if self._sections is not None:
            return self._sections
        path = self.results / "sections.csv"
        if not path.exists():
            from . import structure
            structure.run(self)
        df = read_csv(path)
        out = []
        for i, row in df.iterrows():
            start = parse_time(row["start"])
            end = parse_time(row["end"])
            name = row.get("name", "")
            name = "" if (name is None or (isinstance(name, float) and np.isnan(name))) else str(name).strip()
            out.append(dict(idx=len(out) + 1, start=start, end=min(end, self.duration), name=name))
        for s in out:
            s["label"] = f"{s['idx']:02d}_{safe_name(s['name'])}" if s["name"] else f"{s['idx']:02d}"
            s["title"] = f"{s['idx']:02d} {s['name'] or '(名前なし)'} {fmt_time(s['start'], 1)}〜{fmt_time(s['end'], 1)}"
        self._sections = out
        return out

    def section_of(self, t: float):
        for s in self.sections():
            if s["start"] <= t < s["end"]:
                return s
        return None

    def invalidate_sections(self):
        self._sections = None
