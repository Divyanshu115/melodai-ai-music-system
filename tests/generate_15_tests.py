"""
tests/generate_15_tests.py

Runs 15 test generations using backend.music_generator.MusicGenerator.

- Prints metadata immediately for each run.
- Saves per-variation metadata files (created by the generator).
- Aggregates master metadata to outputs_music_generator/metadata_results.json.

Usage:
    python -m tests.generate_15_tests
"""

import os
import json
import time
from pathlib import Path
import sys

# ensure project root is on path so "backend" imports work when run as module
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.music_generator import MusicGenerator, OUTPUT_DIR

# Developer-provided uploaded file path
EXAMPLE_UPLOADED_FILE_URL = "/mnt/data/f957d385-b1a2-432d-8fbb-7cef8d98d995.png"

TESTS = [
    {"prompt": "Upbeat happy pop music for an ad", "mood": "happy", "energy": 8, "duration": 5},
    {"prompt": "Sad slow piano melody for a short film cue", "mood": "sad", "energy": 2, "duration": 6},
    {"prompt": "Energetic electronic dance music for training", "mood": "energetic", "energy": 9, "duration": 7},
    {"prompt": "Calm ambient background for meditation", "mood": "calm", "energy": 3, "duration": 8},
    {"prompt": "Romantic acoustic guitar bed", "mood": "romantic", "energy": 4, "duration": 6},
    {"prompt": "Intense dramatic orchestral build-up", "mood": "energetic", "energy": 9, "duration": 8},
    {"prompt": "Groovy funk bass loop", "mood": "happy", "energy": 7, "duration": 5},
    {"prompt": "Mysterious dark atmospheric for a game", "mood": "mysterious", "energy": 4, "duration": 6},
    {"prompt": "Chill lofi beats for studying", "mood": "calm", "energy": 3, "duration": 6},
    {"prompt": "Fast techno track for running", "mood": "energetic", "energy": 8, "duration": 5},
    {"prompt": "Epic cinematic trailer sting", "mood": "energetic", "energy": 10, "duration": 6},
    {"prompt": "Soft romantic piano and strings", "mood": "romantic", "energy": 3, "duration": 6},
    {"prompt": "Ambient drone to create tension", "mood": "mysterious", "energy": 2, "duration": 7},
    {"prompt": "Happy ukulele melody for kids", "mood": "happy", "energy": 6, "duration": 5},
    {"prompt": "Minimal percussion loop, focus workspace", "mood": "calm", "energy": 5, "duration": 5}
]

def main():
    gen = MusicGenerator(model_name="facebook/musicgen-small")
    master = {
        "example_uploaded_file_url": EXAMPLE_UPLOADED_FILE_URL,
        "runs": []
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, t in enumerate(TESTS, start=1):
        print("\n" + "="*70)
        print(f"[TEST {idx}/{len(TESTS)}] Prompt: {t['prompt']}")
        start_time = time.time()
        try:
            meta = gen.generate(
                prompt=t["prompt"],
                duration=t["duration"],
                energy=t["energy"],
                mood=t["mood"],
                variations=1,
                retries=3
            )
            elapsed = time.time() - start_time
            # print metadata summary immediately
            print(f"[TEST {idx}] Completed in {elapsed:.2f}s")
            print("Returned metadata (summary):")
            # show concise summary of files
            print("  id:", meta.get("id"))
            print("  model:", meta.get("model"))
            print("  prompt:", (meta.get("prompt") or "")[:140])
            print("  duration_seconds:", meta.get("duration_seconds"))
            print("  temperature:", meta.get("temperature"))
            print("  cfg_coef:", meta.get("cfg_coef"))
            print("  files:")
            for f in meta.get("files", []):
                print("    - variation:", f.get("variation"))
                print("      wav_raw:", f.get("wav_raw"))
                print("      wav_final:", f.get("wav_final"))
                print("      mp3:", f.get("mp3"))
                print("      generation_time_s:", f.get("generation_time_s"))
                print("      size_bytes:", f.get("size_bytes"))

            master["runs"].append({"test_idx": idx, "params": t, "metadata": meta})
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[TEST {idx}] FAILED after {elapsed:.2f}s — error: {e}")
            master["runs"].append({"test_idx": idx, "params": t, "error": str(e)})

    # Save aggregated master metadata
    out_master = os.path.join(OUTPUT_DIR, "metadata_results.json")
    with open(out_master, "w", encoding="utf-8") as fh:
        json.dump(master, fh, indent=2)
    print("\nAll tests done. Master metadata saved to:", out_master)

if __name__ == "__main__":
    main()
