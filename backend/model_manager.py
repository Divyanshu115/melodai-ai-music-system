# backend/model_manager.py
"""
HuggingFace-based multi-model manager for MusicGen (Task 3.3)

Supports:
- facebook/musicgen-small   (fast)
- facebook/musicgen-medium  (balanced)
- facebook/musicgen-large   (best quality)
- facebook/musicgen-melody  (melody conditioning)

Requirements:
    pip install transformers accelerate
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple, Any, List

import torch
import numpy as np
from transformers import MusicgenForConditionalGeneration, AutoProcessor


# Rough mapping: tokens per second of audio (HF docs use max_new_tokens)
# You can tweak this if your durations are off.
TOKENS_PER_SECOND = 50


class HFMusicGenModel:
    """
    Wraps a single HF MusicGen model + processor.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[HFMusicGenModel] Loading {model_name} on {self.device}...")
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

        t1 = time.time()
        print(f"[HFMusicGenModel] Loaded {model_name} in {t1 - t0:.2f}s")

    def generate(
        self,
        prompt: str,
        duration: int = 10,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio for a *single* text prompt.

        Returns:
            (audio_numpy, sample_rate)
        """
        sampling_params = sampling_params or {}

        # Convert desired duration (sec) to max_new_tokens
        max_new_tokens = int(duration * sampling_params.get("tokens_per_second", TOKENS_PER_SECOND))

        temperature = float(sampling_params.get("temperature", 1.0))
        top_k = int(sampling_params.get("top_k", 250))
        top_p = float(sampling_params.get("top_p", 0.95))
        do_sample = bool(sampling_params.get("do_sample", True))

        # Prepare inputs
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

        # HF MusicGen docs: output is (batch, channels, samples)
        # Convert to numpy mono
        audio = audio_values[0, 0].cpu().numpy()
        sample_rate = self.model.config.audio_encoder.sampling_rate

        return audio, sample_rate

    def generate_with_melody(
        self,
        prompt: str,
        melody_waveform: np.ndarray,
        melody_sample_rate: int,
        duration: int = 10,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Melody-conditioned generation (for facebook/musicgen-melody).
        This assumes HF MusicGen supports audio conditioning as per HF examples.
        """
        sampling_params = sampling_params or {}

        max_new_tokens = int(duration * sampling_params.get("tokens_per_second", TOKENS_PER_SECOND))
        temperature = float(sampling_params.get("temperature", 1.0))
        top_k = int(sampling_params.get("top_k", 250))
        top_p = float(sampling_params.get("top_p", 0.95))
        do_sample = bool(sampling_params.get("do_sample", True))

        # Melody conditioning: HF interface may evolve, adjust if HF docs change
        # We'll treat melody as "audio" input into the processor.
        melody_tensor = torch.tensor(melody_waveform).unsqueeze(0)  # (1, samples)
        inputs = self.processor(
            text=[prompt],
            audio=melody_tensor,
            sampling_rate=melody_sample_rate,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

        audio = audio_values[0, 0].cpu().numpy()
        sample_rate = self.model.config.audio_encoder.sampling_rate

        return audio, sample_rate


class ModelManager:
    """
    Manages multiple HF MusicGen models and selects between them.

    Use:
        mgr = ModelManager()
        model_name = mgr.auto_select_model(duration=30, user_choice="facebook/musicgen-medium")
        audio, sr = mgr.generate(model_name, "happy edm", duration=30, sampling_params={...})
    """

    def __init__(self, device: Optional[str] = None):
        self.models: Dict[str, HFMusicGenModel] = {}
        self.current_model_name: Optional[str] = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Known variants mapping
        self.known_models: Dict[str, str] = {
            "small": "facebook/musicgen-small",
            "medium": "facebook/musicgen-medium",
            "large": "facebook/musicgen-large",
            "melody": "facebook/musicgen-melody",
            # direct names also allowed:
            "facebook/musicgen-small": "facebook/musicgen-small",
            "facebook/musicgen-medium": "facebook/musicgen-medium",
            "facebook/musicgen-large": "facebook/musicgen-large",
            "facebook/musicgen-melody": "facebook/musicgen-melody",
        }

    # -------------------------------
    # Resolve model name / alias
    # -------------------------------
    def resolve_model_name(self, name: str) -> str:
        if name in self.known_models:
            return self.known_models[name]
        # if unknown, assume user passed full HF repo id
        return name

    # -------------------------------
    # Load Model (Lazy Loading)
    # -------------------------------
    def load_model(self, model_name: str) -> HFMusicGenModel:
        resolved = self.resolve_model_name(model_name)

        if resolved in self.models:
            self.current_model_name = resolved
            return self.models[resolved]

        print(f"[ModelManager] Loading HF MusicGen model: {resolved}")
        try:
            model = HFMusicGenModel(resolved, device=self.device)
        except Exception as e:
            # Fallback: try small model
            print(f"[ModelManager] Error loading {resolved}: {e}")
            fallback = "facebook/musicgen-small"
            if resolved != fallback:
                print(f"[ModelManager] Falling back to {fallback}")
                model = HFMusicGenModel(fallback, device=self.device)
                resolved = fallback
            else:
                raise

        self.models[resolved] = model
        self.current_model_name = resolved
        return model

    # -------------------------------
    # Auto-select model
    # -------------------------------
    def auto_select_model(
        self,
        duration: int,
        user_choice: Optional[str] = None,
        quality_preference: Optional[str] = None,
    ) -> str:
        """
        Picks model based on:
        - User choice (if any)
        - Duration (long → smaller model)
        - Quality vs speed tradeoff (via quality_preference: "fast", "balanced", "quality")
        """
        if user_choice:
            return self.resolve_model_name(user_choice)

        # Quality preference overrides duration logic
        if quality_preference == "fast":
            return "facebook/musicgen-small"
        if quality_preference == "balanced":
            return "facebook/musicgen-medium"
        if quality_preference == "quality":
            return "facebook/musicgen-large"

        # Duration-based heuristic
        if duration >= 90:
            return "facebook/musicgen-small"
        if duration >= 45:
            return "facebook/musicgen-medium"

        # Shorter clips → use large by default
        return "facebook/musicgen-large"

    # -------------------------------
    # Text-only generation
    # -------------------------------
    def generate(
        self,
        model_name: str,
        prompt: str,
        duration: int = 10,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate a single clip from text prompt.
        Returns (audio_numpy, sample_rate).
        """
        model = self.load_model(model_name)
        audio, sr = model.generate(
            prompt=prompt,
            duration=duration,
            sampling_params=sampling_params,
        )
        return audio, sr

    # -------------------------------
    # Melody-conditioned generation
    # -------------------------------
    def generate_melody_conditioned(
        self,
        prompt: str,
        melody_waveform: np.ndarray,
        melody_sample_rate: int,
        duration: int = 10,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Use 'facebook/musicgen-melody' to condition on audio + text.
        """
        model = self.load_model("facebook/musicgen-melody")
        audio, sr = model.generate_with_melody(
            prompt=prompt,
            melody_waveform=melody_waveform,
            melody_sample_rate=melody_sample_rate,
            duration=duration,
            sampling_params=sampling_params,
        )
        return audio, sr

    # -------------------------------
    # Utility
    # -------------------------------
    def list_loaded_models(self) -> List[str]:
        return list(self.models.keys())
