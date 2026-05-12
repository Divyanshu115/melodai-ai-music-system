"""
backend/music_generator.py

MusicGenerator: wraps MusicGen model, maps high-level params -> model params,
generates audio, applies post-processing (normalize, fade in/out, trim silence),
converts to MP3, returns file paths + metadata.

Dependencies:
    pip install torch transformers numpy soundfile pydub

Notes:
 - pydub requires ffmpeg installed on the machine for MP3 export.
 - If running on CPU only, generation will be slow; prefer GPU.
 - This implementation avoids librosa to reduce dependency issues on Windows.
"""

import os
import time
import uuid
import json
import time
import tempfile
from typing import Dict, Any, Tuple, Optional

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from pydub import AudioSegment, effects, silence

# ----------------- CONFIG -----------------
# Path to optional JSON param config; if missing we'll use defaults
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "params_config.json")

# Output directory (relative to project root)
OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "outputs_music_generator"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default model id (same as earlier tasks)
DEFAULT_MODEL_NAME = "facebook/musicgen-small"

# Approx tokens/sec for MusicGen
TOKENS_PER_SECOND = 50

# Developer-provided uploaded file path (will be stored in metadata)
REFERENCE_IMAGE = "/mnt/data/f957d385-b1a2-432d-8fbb-7cef8d98d995.png"
# ------------------------------------------

class GenerationError(Exception):
    pass


class MusicGenerator:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, config_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the model and config.
        - model_name: huggingface model id for MusicGen.
        - config_path: optional param config; fallback to embedded defaults.
        - device: "cuda" or "cpu" or None (auto-detect)
        """
        self.model_name = model_name
        self.config = self._load_config(config_path or CONFIG_PATH)
        self.device = self._choose_device(device)
        self._processor = None
        self._model = None
        self._init_model()

    # ----------------- Initialization & Config -----------------
    def _load_config(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            # fallback defaults
            return {
                "energy_map": {
                    "min_temp": 0.7,
                    "max_temp": 1.3,
                    "min_cfg": 1.5,
                    "max_cfg": 5.0
                },
                "mood_defaults": {
                    "happy": {"style": "pop", "tempo": "fast", "instruments": ["drums", "guitar", "synth"]},
                    "sad": {"style": "piano", "tempo": "slow", "instruments": ["piano", "strings"]},
                    "calm": {"style": "ambient", "tempo": "slow", "instruments": ["pads", "soft piano"]},
                    "energetic": {"style": "electronic", "tempo": "fast", "instruments": ["synth", "drums", "bass"]},
                    "romantic": {"style": "acoustic", "tempo": "medium", "instruments": ["guitar", "piano", "strings"]},
                    "mysterious": {"style": "ambient horror", "tempo": "slow", "instruments": ["drones", "pads", "strings"]}
                },
                "fade_ms": 1000,
                "silence_trim_db": -40,
                "min_silence_len_ms": 200
            }

    def _choose_device(self, device_hint: Optional[str]) -> torch.device:
        if device_hint:
            if device_hint.lower() == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_model(self):
        if self._model is not None:
            return
        print(f"[MusicGenerator] Loading processor & model: {self.model_name} (device={self.device}) ...")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
        self._model.to(self.device)

        # Ensure pad/eos tokens
        if self._model.config.pad_token_id is None:
            self._model.config.pad_token_id = self._processor.tokenizer.pad_token_id
        if self._model.config.eos_token_id is None:
            self._model.config.eos_token_id = self._processor.tokenizer.eos_token_id

    # ----------------- Parameter Mapping -----------------
    def _map_energy_to_params(self, energy: int) -> Tuple[float, float]:
        """
        Map energy (1..10) to temperature and cfg_coef.
        Returns (temperature, cfg_coef) via linear interpolation.
        """
        try:
            energy = int(energy)
        except Exception:
            energy = 6
        energy = max(1, min(10, energy))
        emap = self.config.get("energy_map", {})
        min_temp = emap.get("min_temp", 0.7)
        max_temp = emap.get("max_temp", 1.3)
        min_cfg = emap.get("min_cfg", 1.5)
        max_cfg = emap.get("max_cfg", 5.0)

        t = min_temp + (energy - 1) / 9.0 * (max_temp - min_temp)
        cfg = min_cfg + (energy - 1) / 9.0 * (max_cfg - min_cfg)
        return float(round(t, 3)), float(round(cfg, 3))

    def _map_mood_to_overrides(self, mood: str) -> Dict[str, Any]:
        mood = (mood or "").strip().lower()
        defaults = self.config.get("mood_defaults", {})
        return defaults.get(mood, {})

    # ----------------- Audio utils / Post-processing -----------------
    def _wav_to_pydub(self, wav_path: str) -> AudioSegment:
        """Load wav file and return pydub AudioSegment."""
        return AudioSegment.from_file(wav_path, format="wav")

    def _save_wav(self, samples: np.ndarray, sr: int, out_path: str):
        """
        Saves numpy float32 (-1..1) or int16 to wav using soundfile.
        Accepts samples shaped (seq,) or (channels, seq) or (seq, channels).
        """
        data = samples
        if samples.ndim == 1:
            data = samples
        elif samples.ndim == 2:
            # If shape is (channels, length) transpose to (length, channels)
            if samples.shape[0] <= 2 and samples.shape[0] != samples.shape[1]:
                data = samples.T
            else:
                data = samples
        sf.write(out_path, data, sr, subtype="PCM_16")

    def _normalize_and_fade_trim(self, wav_path: str, fade_ms: Optional[int] = None, silence_trim_db: Optional[int] = None) -> AudioSegment:
        """
        Load wav -> normalize -> trim silence edges -> fade in/out -> return AudioSegment
        """
        fade_ms = int(fade_ms or self.config.get("fade_ms", 1000))
        silence_trim_db = int(silence_trim_db or self.config.get("silence_trim_db", -40))

        seg = self._wav_to_pydub(wav_path)
        seg = effects.normalize(seg)

        nonsilent = silence.detect_nonsilent(seg, min_silence_len=self.config.get("min_silence_len_ms", 200), silence_thresh=silence_trim_db)
        if nonsilent:
            start_ms = max(0, nonsilent[0][0] - 20)
            end_ms = min(len(seg), nonsilent[-1][1] + 20)
            seg = seg[start_ms:end_ms]

        if len(seg) > fade_ms * 2:
            seg = seg.fade_in(fade_ms).fade_out(fade_ms)
        else:
            small = max(10, int(len(seg) * 0.05))
            seg = seg.fade_in(small).fade_out(small)
        return seg

    # ----------------- Generation (patched to ensure files & metadata) -----------------
    def generate(self, prompt: str, duration: int = 30, energy: int = 6, mood: Optional[str] = None,
                 temperature: Optional[float] = None, top_k: int = 250, top_p: float = 0.95,
                 cfg_coef: Optional[float] = None, variations: int = 1, retries: int = 3) -> Dict[str, Any]:
        """
        Generate music from prompt + params.
        Writes:
          - raw wav: OUTPUT_DIR/{id}_{var}.wav
          - final wav: OUTPUT_DIR/{id}_{var}_final.wav
          - mp3: OUTPUT_DIR/{id}_{var}.mp3
          - per-variation metadata: OUTPUT_DIR/{id}_var{var}_meta.json
          - summary metadata: OUTPUT_DIR/{id}_summary_meta.json

        Returns metadata dict.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be provided")

        # apply mood overrides
        mood_overrides = self._map_mood_to_overrides(mood or "")
        temp_map, cfg_map = self._map_energy_to_params(energy)
        temperature = float(temperature) if temperature is not None else temp_map
        cfg_coef = float(cfg_coef) if cfg_coef is not None else cfg_map

        metadata = {
            "id": uuid.uuid4().hex,
            "model": self.model_name,
            "prompt": prompt,
            "mood": mood,
            "energy": energy,
            "duration_seconds": int(duration),
            "temperature": temperature,
            "cfg_coef": cfg_coef,
            "top_k": int(top_k),
            "top_p": float(top_p),
            "variations": int(max(1, variations)),
            "created_at": time.time(),
            "files": [],
            "generation_attempts": 0,
            "reference_image": REFERENCE_IMAGE
        }

        # augment final prompt with mood hints if available
        final_prompt = prompt
        if mood_overrides:
            parts = [final_prompt]
            if mood_overrides.get("style"):
                parts.append(f"style: {mood_overrides['style']}")
            if mood_overrides.get("instruments"):
                parts.append(f"instruments: {', '.join(mood_overrides['instruments'])}")
            if mood_overrides.get("tempo"):
                parts.append(f"tempo: {mood_overrides['tempo']}")
            final_prompt = ". ".join(parts)

        proc = self._processor
        model = self._model
        device = self.device
        max_new_tokens = int(duration * TOKENS_PER_SECOND)

        # generation loop with retries and per-variation metadata writes
        for var_idx in range(1, metadata["variations"] + 1):
            attempt = 0
            success = False
            last_exc = None
            while attempt < retries and not success:
                attempt += 1
                metadata["generation_attempts"] += 1
                print(f"[MusicGenerator] Generating var {var_idx} attempt {attempt} (prompt preview: {final_prompt[:80]})")
                try:
                    tstart = time.time()
                    inputs = proc(text=[final_prompt], padding=True, return_tensors="pt").to(device)

                    with torch.no_grad():
                        audio_values = model.generate(
                            **inputs,
                            do_sample=True,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            guidance_scale=cfg_coef,
                            max_new_tokens=max_new_tokens,
                        )
                    elapsed = time.time() - tstart

                    audio_tensor = audio_values[0, 0].cpu().numpy()
                    sampling_rate = model.config.audio_encoder.sampling_rate

                    base_name = f"{metadata['id']}_{var_idx}"
                    wav_temp = os.path.join(OUTPUT_DIR, f"{base_name}.wav")
                    self._save_wav(audio_tensor, sampling_rate, wav_temp)

                    # validate raw wav written
                    if not os.path.exists(wav_temp) or os.path.getsize(wav_temp) == 0:
                        raise RuntimeError(f"Raw wav file not written or empty: {wav_temp}")

                    # Post-process: normalize, trim silence, fade
                    seg = self._normalize_and_fade_trim(wav_temp)

                    final_wav = os.path.join(OUTPUT_DIR, f"{base_name}_final.wav")
                    seg.export(final_wav, format="wav")

                    final_mp3 = os.path.join(OUTPUT_DIR, f"{base_name}.mp3")
                    seg.export(final_mp3, format="mp3", bitrate="192k")

                    # sanity checks
                    for p in (final_wav, final_mp3):
                        if not os.path.exists(p) or os.path.getsize(p) == 0:
                            raise RuntimeError(f"Output file missing or empty: {p}")

                    file_meta = {
                        "variation": var_idx,
                        "attempt": attempt,
                        "wav_raw": wav_temp,
                        "wav_final": final_wav,
                        "mp3": final_mp3,
                        "sampling_rate": sampling_rate,
                        "generation_time_s": round(elapsed, 3),
                        "size_bytes": os.path.getsize(final_mp3),
                        "created_at": time.time()
                    }
                    metadata["files"].append(file_meta)

                    # save per-variation metadata immediately
                    meta_path = os.path.join(OUTPUT_DIR, f"{metadata['id']}_var{var_idx}_meta.json")
                    with open(meta_path, "w", encoding="utf-8") as fh:
                        json.dump({"metadata": metadata, "last_file": file_meta}, fh, indent=2)
                    print(f"[MusicGenerator] Saved files: {final_wav} , {final_mp3}")
                    print(f"[MusicGenerator] Saved meta: {meta_path}")
                    success = True
                except Exception as exc:
                    last_exc = exc
                    print(f"[MusicGenerator] Generation attempt {attempt} failed: {exc}")
                    time.sleep(1.0 + attempt * 1.5)
            if not success:
                raise GenerationError(f"Failed to generate variation {var_idx} after {retries} attempts. Last error: {last_exc}")

        metadata["finished_at"] = time.time()
        metadata["elapsed_total_s"] = metadata["finished_at"] - metadata["created_at"]

        # Save overall metadata summary
        summary_path = os.path.join(OUTPUT_DIR, f"{metadata['id']}_summary_meta.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        print(f"[MusicGenerator] Wrote overall metadata: {summary_path}")

        return metadata
