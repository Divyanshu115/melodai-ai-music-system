# backend/music_variations.py
"""
Helpers for advanced music features:
- Variations generation
- Music extension / continuation

Relies on MusicGenerator from music_generator.py
"""

import os
import uuid
import time
from typing import List, Tuple, Dict, Any, Optional

from pydub import AudioSegment

from .music_generator import MusicGenerator, OUTPUT_DIR


# Reuse a single MusicGenerator instance to avoid reloading model repeatedly
_generator_singleton: Optional[MusicGenerator] = None


def _get_generator() -> MusicGenerator:
    global _generator_singleton
    if _generator_singleton is None:
        _generator_singleton = MusicGenerator()
    return _generator_singleton


def generate_variations(
    base_prompt: str,
    num_variations: int = 3,
    duration: int = 30,
    energy: int = 6,
    mood: Optional[str] = None,
    temperature: Optional[float] = None,
    top_k: int = 250,
    top_p: float = 0.95,
    cfg_coef: Optional[float] = None,
    retries: int = 3,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Generate multiple variations of a base prompt.
    Returns (list_of_mp3_paths, full_metadata_dict).
    """

    gen = _get_generator()

    meta = gen.generate(
        prompt=base_prompt,
        duration=duration,
        energy=energy,
        mood=mood,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_coef=cfg_coef,
        variations=max(1, int(num_variations)),
        retries=retries,
    )

    paths: List[str] = []
    for file_meta in meta.get("files", []):
        if isinstance(file_meta, dict):
            p = file_meta.get("mp3") or file_meta.get("wav_final") or file_meta.get("wav_raw")
            if p:
                paths.append(os.path.abspath(p))

    return paths, meta


def extend_music(
    existing_audio_path: str,
    base_prompt: str,
    extension_duration: int = 30,
    energy: int = 6,
    mood: Optional[str] = None,
    temperature: Optional[float] = None,
    top_k: int = 250,
    top_p: float = 0.95,
    cfg_coef: Optional[float] = None,
    retries: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    """
    Extend an existing music clip by generating a continuation and concatenating.
    Returns (new_extended_mp3_path, metadata_dict).
    """

    gen = _get_generator()

    # Generate new continuation segment
    cont_meta = gen.generate(
        prompt=base_prompt,
        duration=extension_duration,
        energy=energy,
        mood=mood,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_coef=cfg_coef,
        variations=1,
        retries=retries,
    )

    cont_path = None
    if cont_meta and isinstance(cont_meta, dict):
        files = cont_meta.get("files") or []
        if files:
            first = files[0]
            if isinstance(first, dict):
                cont_path = first.get("mp3") or first.get("wav_final") or first.get("wav_raw")
            elif isinstance(first, str):
                cont_path = first

    if not cont_path or not os.path.exists(cont_path):
        raise RuntimeError("Continuation generation failed: no valid audio file returned.")

    existing_audio_path = os.path.abspath(existing_audio_path)
    if not os.path.exists(existing_audio_path):
        raise FileNotFoundError(f"Original audio not found: {existing_audio_path}")

    # Concatenate original + continuation
    orig_seg = AudioSegment.from_file(existing_audio_path)
    cont_seg = AudioSegment.from_file(cont_path)

    combined = orig_seg + cont_seg

    out_id = uuid.uuid4().hex
    out_mp3 = os.path.join(OUTPUT_DIR, f"{out_id}_extended.mp3")
    combined.export(out_mp3, format="mp3", bitrate="192k")

    meta = {
        "id": out_id,
        "type": "extension",
        "base_prompt": base_prompt,
        "original_path": existing_audio_path,
        "continuation_path": os.path.abspath(cont_path),
        "extended_path": os.path.abspath(out_mp3),
        "extension_duration_s": int(extension_duration),
        "created_at": time.time(),
    }

    # Save metadata JSON
    meta_path = os.path.join(OUTPUT_DIR, f"{out_id}_extension_meta.json")
    try:
        import json
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    except Exception:
        # best-effort; don't crash UI
        pass

    return os.path.abspath(out_mp3), meta
