# backend/input_processor.py

import os
import time
import json
from typing import Dict, Any

from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()


class InputProcessor:
    """
    Gemini-based Input Processor with:
    - Safe JSON parsing
    - Retry mechanism
    - Streamlit + CLI compatibility
    - Fallback mode
    """

    def __init__(self):

        # ---------------- API KEY LOADING ----------------
        api_key = None

        # Try Streamlit secrets first
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

        # Fallback to .env / system env
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        # Gemini client
        self.client = genai.Client(api_key=api_key)

    # =========================================================
    # MAIN INPUT PROCESSOR
    # =========================================================

    def process_input(self, user_text: str) -> Dict[str, Any]:

        user_text = (user_text or "").strip()

        if not user_text:
            user_text = "Generate calm background music"

        try:
            params = self._extract_with_gemini(user_text)

        except Exception as e:
            print(f"[InputProcessor] Gemini failed, using fallback. Error: {e}")
            params = self._fallback(user_text)

        params["raw_text"] = user_text

        return params

    # =========================================================
    # GEMINI EXTRACTION
    # =========================================================

    def _extract_with_gemini(self, text: str) -> Dict[str, Any]:

        prompt = f"""
Return ONLY valid JSON.
Do NOT explain anything.

{{
  "mood": "happy|sad|calm|energetic|romantic|neutral",
  "energy": 1-10,
  "style": "genre",
  "tempo": "slow|medium|fast",
  "instruments": ["piano","guitar","drums"],
  "context": "general"
}}

User request: {text}
"""

        MAX_RETRIES = 5

        for attempt in range(MAX_RETRIES):

            try:

                response = self.client.models.generate_content(
                    model="models/gemini-2.5-flash",
                    contents=prompt
                )

                return self._safe_json(response.text)

            except Exception as e:

                print(f"[Gemini Retry {attempt + 1}] Failed: {e}")

                # Exponential backoff
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e

    # =========================================================
    # SAFE JSON PARSER
    # =========================================================

    def _safe_json(self, text: str) -> Dict[str, Any]:

        text = text.strip()

        # Remove markdown code fences if Gemini adds them
        if text.startswith("```"):
            text = (
                text.replace("```json", "")
                .replace("```", "")
                .strip()
            )

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON found in Gemini response")

        return json.loads(text[start:end + 1])

    # =========================================================
    # FALLBACK SYSTEM
    # =========================================================

    def _fallback(self, text: str) -> Dict[str, Any]:

        t = text.lower()

        # ---------------- MOOD DETECTION ----------------

        if "calm" in t or "relax" in t or "meditation" in t:
            mood, energy = "calm", 3

        elif "energetic" in t or "workout" in t or "gym" in t:
            mood, energy = "energetic", 8

        elif "sad" in t or "emotional" in t:
            mood, energy = "sad", 4

        elif "happy" in t or "party" in t:
            mood, energy = "happy", 7

        elif "focus" in t or "study" in t:
            mood, energy = "focused", 5

        elif "stress" in t or "anxious" in t:
            mood, energy = "anxious", 6

        else:
            mood, energy = "neutral", 6

        # ---------------- TEMPO ----------------

        if energy <= 3:
            tempo = "slow"
        elif energy >= 7:
            tempo = "fast"
        else:
            tempo = "medium"

        # ---------------- STYLE MAP ----------------

        style_map = {
            "calm": "ambient",
            "energetic": "electronic dance",
            "sad": "piano emotional",
            "happy": "upbeat pop",
            "focused": "lofi",
            "anxious": "tense cinematic",
            "neutral": "instrumental",
        }

        style = style_map.get(mood, "instrumental")

        # ---------------- INSTRUMENT MAP ----------------

        instrument_map = {
            "calm": ["piano", "pads", "strings"],
            "energetic": ["synth", "drums", "bass"],
            "sad": ["piano", "violin"],
            "happy": ["guitar", "drums", "bass"],
            "focused": ["piano", "soft beats"],
            "anxious": ["strings", "synth"],
            "neutral": ["piano"],
        }

        instruments = instrument_map.get(mood, ["piano"])

        # ---------------- CONTEXT PROMPT ----------------

        context = (
            f"{style} {mood} music with "
            f"{', '.join(instruments)}, "
            f"{tempo} tempo, "
            f"high quality immersive sound"
        )

        # ---------------- FINAL OUTPUT ----------------

        return {
            "mood": mood,
            "energy": energy,
            "style": style,
            "tempo": tempo,
            "instruments": instruments,
            "context": context,
        }