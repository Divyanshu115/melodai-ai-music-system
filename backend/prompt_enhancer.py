"""
backend/prompt_enhancer.py

Enhances simple structured parameters (from InputProcessor) into rich
descriptive text prompts suitable for music generation models.

Functions / classes:
 - PromptEnhancer(mood_json_path=None, seed=None)
 - PromptEnhancer.enhance(params, variations=1)
 - PromptEnhancer.validate_prompt(prompt)  # simple checks
"""

import json
import random
import re
from typing import Dict, Any, List, Optional

DEFAULT_MOOD_TEMPLATES = {
    # fallback templates in case json not loaded
    "happy": "upbeat, cheerful, major key, bright instruments, energetic rhythm, positive vibes, {tempo}",
    "sad": "melancholic, slow tempo, minor key, emotional, piano and strings, reflective, somber mood",
    "energetic": "high energy, fast-paced, driving beat, intense rhythm, powerful, dynamic, {tempo}",
    "calm": "soft ambient pads, slow tempo, gentle dynamics, soothing textures, spacious reverb, {tempo}",
    "romantic": "warm timbre, gentle acoustic instruments, slow tempo, expressive melodies, strings and piano, {tempo}",
    # add more here as needed
}


class PromptEnhancer:
    def __init__(self, mood_json_path: Optional[str] = None, seed: Optional[int] = None):
        """
        Load mood templates (JSON) if provided, else use defaults.
        Optionally set random seed for reproducible variations.
        """
        if mood_json_path:
            try:
                with open(mood_json_path, "r", encoding="utf-8") as fh:
                    self.mood_templates = json.load(fh)
            except Exception:
                self.mood_templates = DEFAULT_MOOD_TEMPLATES
        else:
            self.mood_templates = DEFAULT_MOOD_TEMPLATES

        if seed is not None:
            random.seed(seed)

        # small genre/instrument synonyms to spice prompts
        self.genre_adjectives = {
            "pop": ["radio-friendly", "catchy", "bright", "anthemic"],
            "electronic": ["synth-driven", "glossy electronic texture", "pulsing synths"],
            "ambient": ["ethereal", "atmospheric", "soundscape-like"],
            "lofi": ["warm lo-fi tape effect", "dusty vinyl crackle", "chill beat"],
            "orchestral": ["cinematic strings", "heroic brass", "lush orchestration"],
            "acoustic": ["organic guitar", "intimate acoustic performance"],
            "default": ["well-produced", "clean arrangement"]
        }

        # small safety / validation parameters
        self.min_length = 30   # characters
        self.max_length = 600  # characters

    def _pick_genre_adjective(self, style: Optional[str]) -> str:
        if not style:
            return ""
        style_key = style.lower().split()[0]
        return random.choice(self.genre_adjectives.get(style_key, self.genre_adjectives["default"]))

    def _instruments_str(self, instruments: Optional[List[str]]) -> str:
        if not instruments:
            return ""
        # join instruments and add short descriptors
        items = []
        for inst in instruments:
            inst = inst.strip()
            if not inst:
                continue
            # add small descriptor heuristics
            if "piano" in inst.lower():
                items.append("soft piano")
            elif "guitar" in inst.lower():
                items.append("acoustic guitar")
            elif "synth" in inst.lower() or "synth" in inst.lower():
                items.append("synth pads and leads")
            else:
                items.append(inst)
        return ", ".join(items)

    def _tempo_str(self, tempo_param: Optional[Any]) -> str:
        if not tempo_param:
            return ""
        # accept numeric or "slow/medium/fast" text
        t = str(tempo_param).strip()
        if t.isdigit():
            return f"{t} BPM"
        # map words to BPM hints
        map_bpm = {"slow": "60-80 BPM", "medium": "90-110 BPM", "fast": "120-150 BPM"}
        return map_bpm.get(t.lower(), t)

    def _structure_hints(self):
        # short list of structure hints to add variety
        hints = [
            "intro -> verse -> chorus structure",
            "build-up and a dynamic climax at the end",
            "loopable 8-bar phrases with variation",
            "long ambient pads and gradual evolution"
        ]
        return random.choice(hints)

    def enhance(self, params: Dict[str, Any], variations: int = 1) -> List[str]:
        """
        params: dict with keys like mood, energy, style, tempo, instruments, context, raw_text
        variations: how many prompt variations to return
        returns: list of enhanced prompt strings
        """
        results = []
        mood = (params.get("mood") or "").lower()
        energy = params.get("energy")
        style = params.get("style") or ""
        tempo = params.get("tempo") or ""
        instruments = params.get("instruments") or []
        context = params.get("context") or ""
        raw_text = params.get("raw_text") or ""

        # base template
        template = self.mood_templates.get(mood) or self.mood_templates.get("default", "")

        for i in range(max(1, variations)):
            # vary adjectives and descriptors
            genre_adj = self._pick_genre_adjective(style)
            inst_str = self._instruments_str(instruments)
            tempo_str = self._tempo_str(tempo)

            # energy text
            energy_text = ""
            if isinstance(energy, (int, float)):
                if energy >= 8:
                    energy_text = "very high energy"
                elif energy >= 5:
                    energy_text = "moderate energy"
                else:
                    energy_text = "low energy, gentle dynamics"

            # Combine pieces
            parts = []

            # core mood template with tempo placeholder
            mood_text = template.format(tempo=tempo_str) if "{tempo" in template else template
            if mood_text:
                parts.append(mood_text)

            if genre_adj:
                parts.append(genre_adj)

            if inst_str:
                parts.append(f"featuring {inst_str}")

            if energy_text:
                parts.append(energy_text)

            if context:
                parts.append(f"for {context}")

            # structural hint
            parts.append(self._structure_hints())

            # brief instruction to model about length and format
            parts.append("Duration around 10-30 seconds, stereo, clear mix, no vocals (instrumental).")

            # ensure we reference the user's raw text to preserve intent
            parts.append(f"User request: \"{raw_text}\"")

            prompt = ". ".join([p.strip() for p in parts if p]).strip()

            # sometimes add little variation like production terms
            extras = ["use subtle reverb", "add gentle sidechain to the bass", "keep a warm midrange"]
            if random.random() < 0.4:
                prompt += f". {random.choice(extras)}."

            # Trim / validate
            prompt = self._clean_whitespace(prompt)
            if len(prompt) > self.max_length:
                prompt = prompt[: self.max_length]

            # fallback length enforcement
            if len(prompt) < self.min_length:
                prompt = (prompt + ". " + "Provide clear melodic motif.").strip()

            results.append(prompt)

        return results

    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Basic validation checks:
         - length range
         - banned characters / repetitive tokens
         - coherence heuristics (very simple)
        Returns dict: {"ok": bool, "reasons": [..], "length": int}
        """
        reasons = []
        length = len(prompt)

        if length < self.min_length:
            reasons.append("too_short")
        if length > self.max_length:
            reasons.append("too_long")

        # no control characters
        if re.search(r"[\x00-\x08\x0B-\x1F]", prompt):
            reasons.append("invalid_chars")

        # simple duplicate word check (if same word repeated > 8 times)
        words = re.findall(r"\w+", prompt.lower())
        for w in set(words):
            if words.count(w) > 8:
                reasons.append(f"repeated_word:{w}")

        ok = len(reasons) == 0
        return {"ok": ok, "reasons": reasons, "length": length}

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        return re.sub(r"\s{2,}", " ", text).strip()
