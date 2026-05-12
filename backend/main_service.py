"""
backend/main_service.py

Orchestrator for Task 1.6:
 - generate_music_pipeline(user_input) : main pipeline
 - CLI runner to execute N tests (10 by default)
 - Logging, error recovery, config management
"""


from backend.quality_scorer import QualityScorer, default_meta_to_audio_path
from backend.cache_manager import CacheManager
cache = CacheManager()


import os
import json
import time
import logging
from typing import Tuple, Dict, Any, Optional, List

import soundfile as sf  # for saving HF audio to WAV

from backend.model_manager import ModelManager
from backend.input_processor import InputProcessor
from backend.prompt_enhancer import PromptEnhancer
from backend.music_generator import MusicGenerator


# HF multi-model manager (Task 3.3)
model_manager = ModelManager()

# Load config
_CFG_PATH = os.path.join(os.path.dirname(__file__), "main_config.json")
with open(_CFG_PATH, "r", encoding="utf-8") as fh:
    CONFIG = json.load(fh)

# Setup outputs dir (absolute)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), CONFIG.get("outputs_dir", "../outputs_main_service"))
)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Setup logging
LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), CONFIG["logging"]["file"]))
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=getattr(logging, CONFIG["logging"].get("level", "INFO")),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main_service")

# Initialize components lazily
_OPENAI_KEY = os.getenv(CONFIG.get("openai_api_key_env", "OPENAI_API_KEY"), "").strip()
_INPUT_PROCESSOR = InputProcessor()
_PROMPT_ENHANCER = PromptEnhancer(
    mood_json_path=os.path.join(os.path.dirname(__file__), "params_config.json"),
    seed=42,
)
_MUSIC_GENERATOR = MusicGenerator(model_name=CONFIG.get("model_name"))


def _build_sampling_overrides_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper: Build HF sampling params dict from pipeline params.
    Used by HF ModelManager.
    """
    return {
        "temperature": params.get("temperature", 1.0),
        "top_k": params.get("top_k", 250),
        "top_p": params.get("top_p", 0.95),
        "cfg_coef": params.get("cfg_coef", 3.0),
        "tokens_per_second": CONFIG.get("tokens_per_second", 50),
    }


def _save_hf_audio_to_meta(
    audio_np,
    sample_rate: int,
    variation_idx: int,
) -> Dict[str, Any]:
    """
    Save HF-generated numpy audio array to WAV in OUTPUTS_DIR
    and return a minimal metadata dict compatible with the rest of the app.
    """
    ts = int(time.time())
    base_name = f"hf_musicgen_{ts}_{variation_idx}"
    wav_path = os.path.join(OUTPUTS_DIR, f"{base_name}.wav")

    sf.write(wav_path, audio_np, sample_rate)

    meta = {
        "id": base_name,
        "files": [
            {
                "wav_raw": wav_path,
            }
        ],
    }
    return meta


def generate_music_pipeline(
    user_input: str,
    duration: Optional[int] = None,
    variations: Optional[int] = None,
    retries: Optional[int] = None,
    debug: bool = False,
    # --- Task 3.3 new knobs ---
    model_name: Optional[str] = None,
    quality_preference: Optional[str] = None,
    sampling_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], List[str]]:
    """
    Full pipeline:
      1) input -> structured params (InputProcessor)
      2) params -> enhanced prompt(s) (PromptEnhancer)
      3) generate audio (HF ModelManager first, fallback to MusicGenerator)

    Returns:
      - metadata (dict) returned by generator (or failure meta)
      - params (dict) produced by InputProcessor (completed with defaults)
      - enhanced_prompts (list) produced by PromptEnhancer
    """
    duration = int(duration or CONFIG.get("default_duration", 6))
    variations = int(variations or CONFIG.get("default_variations", 1))
    retries = int(retries or CONFIG.get("default_retries", 3))

    logger.info("Pipeline start — user_input: %s", user_input[:140])
    params: Dict[str, Any] = {}
    enhanced_prompts: List[str] = []

    # --- Stage 1: Input Processing (with recovery) ---
    try:
        params = _INPUT_PROCESSOR.process_input(user_input)
        params.setdefault("duration", duration)
        params.setdefault("variations", variations)
        params.setdefault("retries", retries)
        logger.info(
            "InputProcessor result: %s",
            {
                k: params.get(k)
                for k in ["mood", "energy", "style", "tempo", "instruments", "context", "duration"]
            },
        )
    except Exception as e:
        logger.exception("InputProcessor failed: %s", e)
        params = {
            "mood": "neutral",
            "energy": 6,
            "style": "generic",
            "tempo": "medium",
            "instruments": ["piano", "drums"],
            "context": "general",
            "duration": duration,
            "variations": variations,
            "retries": retries,
        }
        logger.info("Falling back to default params: %s", params)

    # Attach optional user model / quality preferences into params
    if model_name:
        params["model_name"] = model_name
    if quality_preference:
        params["quality_preference"] = quality_preference
    if sampling_overrides:
        params.setdefault("sampling_overrides", {}).update(sampling_overrides)

    # --- Stage 2: Prompt Enhancement (with recovery) ---
    try:
        enhanced_prompts = _PROMPT_ENHANCER.enhance(params, variations=variations)
        logger.info("Generated %d enhanced prompt(s).", len(enhanced_prompts))
        if debug:
            for i, p in enumerate(enhanced_prompts, 1):
                logger.debug("Prompt %d: %s", i, p[:300])
    except Exception as e:
        logger.exception("PromptEnhancer failed: %s", e)
        base = params.get("raw_text") or user_input
        enhanced_prompts = [
            f"{base}. mood: {params.get('mood')}. instruments: {', '.join(params.get('instruments', []))}"
        ] * variations
        logger.info("Falling back to basic prompt(s).")

    # --- Stage 3: Music generation (HF ModelManager first, then fallback) ---
    result_meta: Optional[Dict[str, Any]] = None
    last_exc: Optional[Exception] = None

    # Decide which model to use overall (can be adjusted per variation if needed)
    chosen_model_name = model_manager.auto_select_model(
        duration=params.get("duration", duration),
        user_choice=params.get("model_name"),
        quality_preference=params.get("quality_preference"),
    )
    params["model_used"] = chosen_model_name  # propagate into params/meta

    for i, prompt in enumerate(enhanced_prompts, start=1):
        logger.info("Generating variation %d/%d", i, len(enhanced_prompts))

        # Sampling params for HF
        hf_sampling = sampling_overrides or params.get("sampling_overrides") or _build_sampling_overrides_from_params(
            params
        )

        try:
            # ---- Try Hugging Face MusicGen via ModelManager (Task 3.3) ----
            logger.info("Using HF ModelManager with model: %s", chosen_model_name)
            audio_np, sample_rate = model_manager.generate(
                model_name=chosen_model_name,
                prompt=prompt,
                duration=params.get("duration", duration),
                sampling_params=hf_sampling,
            )

            meta = _save_hf_audio_to_meta(audio_np, sample_rate, variation_idx=i)
            meta["model_used"] = chosen_model_name

            # Attach pipeline info
            meta["pipeline"] = {
                "user_input": user_input,
                "enhanced_prompt": prompt,
                "params": params,
                "reference_image": CONFIG.get("reference_image"),
            }

            result_meta = meta
            logger.info(
                "HF MusicGen generation succeeded: id=%s files=%d",
                meta.get("id"),
                len(meta.get("files", [])),
            )
            break

        except Exception as e:
            # If HF path fails, log and try legacy MusicGenerator as fallback
            last_exc = e
            logger.exception("HF MusicGen failed for variation %d: %s", i, e)
            logger.info("Falling back to legacy MusicGenerator for this variation...")

            try:
                meta = _MUSIC_GENERATOR.generate(
                    prompt=prompt,
                    duration=params.get("duration", duration),
                    energy=params.get("energy", 6),
                    mood=params.get("mood"),
                    temperature=params.get("temperature", None),
                    top_k=params.get("top_k", 250),
                    top_p=params.get("top_p", 0.95),
                    cfg_coef=params.get("cfg_coef", None),
                    variations=1,
                    retries=retries,
                )
                meta["pipeline"] = {
                    "user_input": user_input,
                    "enhanced_prompt": prompt,
                    "params": params,
                    "reference_image": CONFIG.get("reference_image"),
                }
                meta["model_used"] = CONFIG.get("model_name")
                result_meta = meta
                logger.info(
                    "Legacy MusicGenerator succeeded: id=%s files=%d",
                    meta.get("id"),
                    len(meta.get("files", [])),
                )
                break
            except Exception as e2:
                last_exc = e2
                logger.exception("MusicGenerator fallback failed for variation %d: %s", i, e2)
                continue

    if result_meta is None:
        logger.error("All generation attempts failed for user_input: %s", user_input[:120])
        result_meta = {
            "id": None,
            "error": str(last_exc),
            "pipeline": {
                "user_input": user_input,
                "params": params,
                "enhanced_prompts": enhanced_prompts,
                "reference_image": CONFIG.get("reference_image"),
            },
        }

    logger.info("Pipeline finished for input (preview): %s -> %s", user_input[:80], result_meta.get("id"))
    return result_meta, params, enhanced_prompts
    cache_key = cache.get_cache_key(prompt=user_input, params=params)
    cached_result = cache.get(cache_key)

    if cached_result:
     return cached_result, params, [prompt]



# -------------------- CLI test runner (10 inputs) --------------------
DEFAULT_TEST_INPUTS = [
    "Upbeat pop jingle for a 10 second ad",
    "Slow piano piece for emotional scene",
    "Energetic EDM loop for gym motivation",
    "Soft ambient background for meditation",
    "Romantic guitar bed for wedding montage",
    "Dark and eerie tension for a horror level",
    "Funky bass loop for a short trailer",
    "Lo-fi beats for studying and concentration",
    "Cinematic orchestra hit for dramatic reveal",
    "Minimal percussion loop for productivity",
]


def run_end_to_end_tests(inputs: Optional[List[str]] = None, save_master: bool = True) -> Dict[str, Any]:
    inputs = inputs or DEFAULT_TEST_INPUTS[:10]
    master = {
        "runs": [],
        "created_at": time.time(),
        "reference_image": CONFIG.get("reference_image"),
    }

    for i, user_in in enumerate(inputs, start=1):
        logger.info("=== TEST %d/%d START: %s", i, len(inputs), user_in[:140])
        start = time.time()
        try:
            meta, params, enhanced = generate_music_pipeline(
                user_in,
                duration=CONFIG.get("default_duration"),
                variations=CONFIG.get("default_variations"),
                retries=CONFIG.get("default_retries"),
            )
            elapsed = time.time() - start
            logger.info("=== TEST %d COMPLETED in %.2fs — id=%s", i, elapsed, meta.get("id"))
            master["runs"].append(
                {
                    "test_idx": i,
                    "input": user_in,
                    "elapsed_s": round(elapsed, 3),
                    "meta": meta,
                    "params": params,
                }
            )
        except Exception as e:
            elapsed = time.time() - start
            logger.exception("=== TEST %d FAILED after %.2fs: %s", i, elapsed, e)
            master["runs"].append(
                {
                    "test_idx": i,
                    "input": user_in,
                    "elapsed_s": round(elapsed, 3),
                    "error": str(e),
                }
            )

    if save_master:
        out_path = os.path.join(OUTPUTS_DIR, f"main_service_test_results_{int(time.time())}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(master, fh, indent=2)
        logger.info("Saved master test results to: %s", out_path)

    return master


# CLI entrypoint
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Main service: run pipeline or tests")
    parser.add_argument("--run-tests", action="store_true", help="Run end-to-end test inputs (10 by default)")
    parser.add_argument("--single", type=str, default=None, help="Run pipeline once with the provided user text")
    parser.add_argument("--list-inputs", action="store_true", help="Show default test inputs and exit")
    args = parser.parse_args()

    if args.list_inputs:
        for i, s in enumerate(DEFAULT_TEST_INPUTS, start=1):
            print(f"{i}. {s}")
        raise SystemExit(0)

    if args.run_tests:
        run_end_to_end_tests()
        raise SystemExit(0)

    if args.single:
        meta, params, enhanced = generate_music_pipeline(args.single, duration=CONFIG.get("default_duration"))
        print("Result metadata (summary):")
        print(
            json.dumps(
                {"id": meta.get("id"), "files": meta.get("files"), "error": meta.get("error", None)},
                indent=2,
            )
        )
        raise SystemExit(0)

    parser.print_help()


# -------------------- Quality-scored wrapper (Task 3.1) --------------------

scorer = QualityScorer()


def generate_music_pipeline_with_quality(**kwargs):
    """
    Wraps your existing generate_music_pipeline with auto quality + retry.
    """

    def _gen():
        # Call your existing pipeline
        return generate_music_pipeline(**kwargs)

    expected_params = {
        "duration": kwargs.get("duration"),
        # If your pipeline extracts mood/energy, you can plug them here
        # "mood": some_mood,
        # "energy": some_energy,
    }

    (meta, params, enhanced_prompts), score_report = scorer.generate_with_quality(
        generator_fn=_gen,
        get_audio_path=default_meta_to_audio_path,
        expected_params=expected_params,
        max_retries=2,
    )

    if isinstance(params, dict):
        params = dict(params)
        params["quality_metrics"] = score_report

    return meta, params, enhanced_prompts
