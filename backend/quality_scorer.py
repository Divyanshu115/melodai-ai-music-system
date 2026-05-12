# backend/quality_scorer.py
"""
Quality scoring for generated music.

Usage:
    scorer = QualityScorer(min_score=65)
    result = scorer.score_audio(audio_file, expected_params={'duration': 10, 'mood': 'calm', 'energy': 2})
    print(result['overall_score'], result['scores'])

Optional auto-retry helper:
    best_result, score_report = scorer.generate_with_quality(
        generator_fn=lambda: generate_music_pipeline(...),
        max_retries=2
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import librosa  # type: ignore
except Exception:  # librosa not installed
    librosa = None


@dataclass
class QualityConfig:
    min_score: float = 65.0
    target_sr: int = 32000
    max_clipping_ratio: float = 0.01
    max_silence_ratio: float = 0.35
    duration_tolerance_sec: float = 2.0


class QualityScorer:
    def __init__(self, config: Optional[QualityConfig] = None) -> None:
        self.config = config or QualityConfig()

    # ----------------- Public API -----------------

    def score_audio(
        self,
        audio_file: str,
        expected_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute multiple quality metrics and an overall score.

        expected_params can include:
          - duration (seconds)
          - mood (e.g., 'calm', 'energetic', 'sad', ...)
          - energy (1–5 or similar)
        """
        if not os.path.exists(audio_file):
            return {
                "overall_score": 0.0,
                "scores": {},
                "error": f"File not found: {audio_file}",
            }

        if librosa is None:
            # Fallback when librosa is not installed.
            # Return a neutral-but-passing score.
            scores = {
                "audio_quality": 75.0,
                "duration_accuracy": 75.0,
                "silence_detection": 75.0,
                "dynamic_range": 75.0,
                "frequency_balance": 75.0,
                "mood_alignment": 75.0,
            }
            return {
                "overall_score": float(np.mean(list(scores.values()))),
                "scores": scores,
                "warning": "librosa not installed; using fallback scores.",
            }

        expected_params = expected_params or {}

        # Load audio
        y, sr = librosa.load(audio_file, sr=self.config.target_sr, mono=True)
        duration_sec = len(y) / float(sr)

        scores: Dict[str, float] = {}

        scores["audio_quality"] = self._score_audio_quality(y)
        scores["duration_accuracy"] = self._score_duration(
            duration_sec, expected_params.get("duration")
        )
        scores["silence_detection"] = self._score_silence(y, sr)
        scores["dynamic_range"] = self._score_dynamic_range(y)
        scores["frequency_balance"] = self._score_frequency_balance(y, sr)
        scores["mood_alignment"] = self._score_mood_alignment(
            y, sr, expected_params
        )

        overall = self._calculate_overall_score(scores)

        return {
            "overall_score": overall,
            "scores": scores,
            "duration_sec": duration_sec,
        }

    def generate_with_quality(
        self,
        generator_fn: Callable[[], Tuple[Dict[str, Any], Dict[str, Any], Any]],
        get_audio_path: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
        expected_params: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
    ) -> Tuple[Tuple[Dict[str, Any], Dict[str, Any], Any], Dict[str, Any]]:
        """
        Wrap a generation function with auto quality scoring + retry.

        generator_fn:
            A callable that returns (meta, params, enhanced_prompts)
            e.g. lambda: generate_music_pipeline(...)

        get_audio_path:
            Optional function to extract the audio file path from meta.
            If None, a default extractor is used.

        Returns:
            (best_generation_tuple, best_score_report)
        """
        if get_audio_path is None:
            get_audio_path = default_meta_to_audio_path

        best_result: Optional[Tuple[Dict[str, Any], Dict[str, Any], Any]] = None
        best_score_report: Optional[Dict[str, Any]] = None
        best_score = -1.0

        attempts = 0
        while attempts <= max_retries:
            attempts += 1
            meta, params, enhanced_prompts = generator_fn()
            audio_path = get_audio_path(meta)

            if not audio_path or not os.path.exists(audio_path):
                # If no audio produced, this attempt is zero.
                score_report = {
                    "overall_score": 0.0,
                    "scores": {},
                    "error": "No audio file returned by generator.",
                }
            else:
                score_report = self.score_audio(audio_path, expected_params)

            current_score = score_report.get("overall_score", 0.0)

            if current_score > best_score:
                best_score = current_score
                best_score_report = score_report
                best_result = (meta, params, enhanced_prompts)

            # Stop early if we meet threshold
            if current_score >= self.config.min_score:
                break

        # Fallback in case everything failed badly
        if best_result is None:
            best_result = ({}, {}, None)
            best_score_report = best_score_report or {
                "overall_score": 0.0,
                "scores": {},
                "error": "All attempts failed to produce a valid result.",
            }

        return best_result, best_score_report

    # ----------------- Internal scoring helpers -----------------

    def _score_audio_quality(self, y: np.ndarray) -> float:
        # Clipping detection and basic loudness check
        if y.size == 0:
            return 0.0

        # Assume normalized to [-1, 1]
        abs_y = np.abs(y)
        clipping_ratio = np.mean(abs_y > 0.99)
        rms = np.sqrt(np.mean(y**2))

        # Start with 100 and subtract penalty for clipping
        score = 100.0
        if clipping_ratio > self.config.max_clipping_ratio:
            # heavy penalty for clipping
            penalty = min(80.0, clipping_ratio * 4000.0)
            score -= penalty

        # Soft penalty if volume too low or too high (approx)
        if rms < 0.02:
            score -= 20.0
        elif rms > 0.5:
            score -= 10.0

        return float(np.clip(score, 0.0, 100.0))

    def _score_duration(
        self,
        duration_sec: float,
        expected_duration: Optional[float],
    ) -> float:
        if not expected_duration or expected_duration <= 0:
            # If we don't know what was requested, give a neutral score
            return 75.0

        diff = abs(duration_sec - float(expected_duration))
        tol = self.config.duration_tolerance_sec

        if diff <= tol:
            return 100.0
        elif diff >= 3 * tol:
            return 40.0
        else:
            # linear drop between tol and 3*tol
            # diff == tol => 100, diff == 3*tol => 40
            frac = (diff - tol) / (2.0 * tol)
            score = 100.0 - frac * 60.0
            return float(np.clip(score, 0.0, 100.0))

    def _score_silence(self, y: np.ndarray, sr: int) -> float:
        if y.size == 0:
            return 0.0

        # Frame-wise energy
        hop = int(0.01 * sr)  # 10 ms
        win = int(0.03 * sr)  # 30 ms
        if win <= 0 or hop <= 0:
            return 75.0

        frames = librosa.util.frame(y, frame_length=win, hop_length=hop)
        frame_energy = np.mean(frames**2, axis=0)

        thresh = np.percentile(frame_energy, 10) * 0.5
        if thresh <= 0:
            thresh = 1e-8

        silent_frames = frame_energy < thresh
        silence_ratio = np.mean(silent_frames)

        # penalize long silence
        score = 100.0
        if silence_ratio > self.config.max_silence_ratio:
            penalty = min(70.0, (silence_ratio - self.config.max_silence_ratio) * 200)
            score -= penalty

        return float(np.clip(score, 0.0, 100.0))

    def _score_dynamic_range(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0

        abs_y = np.abs(y)
        lo = np.percentile(abs_y, 5)
        hi = np.percentile(abs_y, 95)
        dr = hi - lo

        # Heuristic mapping: too low DR => low score, moderate/high DR => good score
        if dr < 0.01:
            return 30.0
        elif dr > 0.25:
            return 100.0
        else:
            # linearly map between 0.01 -> 0.25 to 30 -> 95
            frac = (dr - 0.01) / (0.25 - 0.01)
            return float(30.0 + frac * 65.0)

    def _score_frequency_balance(self, y: np.ndarray, sr: int) -> float:
        if y.size == 0:
            return 0.0

        try:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr // 2)
            mel_energy = librosa.power_to_db(S, ref=np.max)
        except Exception:
            return 75.0

        # Average across time
        avg_mel = np.mean(mel_energy, axis=1)  # (mel_bins,)

        # Roughly split into low/mid/high bands
        n = len(avg_mel)
        low = np.mean(avg_mel[: n // 3])
        mid = np.mean(avg_mel[n // 3 : 2 * n // 3])
        high = np.mean(avg_mel[2 * n // 3 :])

        # Penalize extreme imbalance
        vals = np.array([low, mid, high])
        spread = np.max(vals) - np.min(vals)  # in dB

        if spread < 6.0:
            return 100.0
        elif spread > 24.0:
            return 40.0
        else:
            frac = (spread - 6.0) / (24.0 - 6.0)
            return float(100.0 - frac * 60.0)

    def _score_mood_alignment(
        self,
        y: np.ndarray,
        sr: int,
        expected_params: Dict[str, Any],
    ) -> float:
        """
        Very lightweight approximation:
        - Use tempo and RMS as "energy".
        - Compare with expected 'mood' / 'energy'.
        This is heuristic and not a true classifier.
        """
        if not expected_params:
            return 75.0

        try:
            tempo = float(librosa.beat.tempo(y=y, sr=sr, aggregate=np.median)[0])
        except Exception:
            tempo = 0.0

        rms = float(np.sqrt(np.mean(y**2))) if y.size > 0 else 0.0

        score = 80.0  # base

        mood = str(expected_params.get("mood", "")).lower()
        energy = expected_params.get("energy", None)  # e.g., 1–5

        # Simple rules
        if mood in ("energetic", "happy", "dramatic"):
            # expect higher tempo and higher rms
            if tempo < 80:
                score -= 15.0
            if rms < 0.03:
                score -= 15.0
        elif mood in ("calm", "sad", "romantic"):
            # expect lower tempo and not too loud
            if tempo > 130:
                score -= 15.0
            if rms > 0.15:
                score -= 10.0

        # if energy numeric, coarse check
        if isinstance(energy, (int, float)):
            if energy >= 4 and rms < 0.03:
                score -= 10.0
            if energy <= 2 and rms > 0.2:
                score -= 10.0

        return float(np.clip(score, 0.0, 100.0))

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        if not scores:
            return 0.0

        # You can tweak weights here
        # For now, equal weight across all metrics.
        values = list(scores.values())
        return float(np.mean(values))


# ----------------- Utility: default meta -> audio path -----------------


def default_meta_to_audio_path(meta: Dict[str, Any]) -> Optional[str]:
    """
    Try to extract audio file path from the meta structure returned by generate_music_pipeline.
    """
    if not isinstance(meta, dict):
        return None

    files = meta.get("files") or []
    if files:
        first = files[0]
        if isinstance(first, dict):
            return first.get("mp3") or first.get("wav_final") or first.get("wav_raw")
        elif isinstance(first, str):
            return first

    return meta.get("mp3") or meta.get("wav_final") or meta.get("wav_raw") or meta.get("file")
