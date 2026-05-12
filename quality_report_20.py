# quality_report_20.py
# Generate quality score report for up to 20 audio samples (Task 3.1)

import os
import glob
import json
from statistics import mean

from backend.quality_scorer import QualityScorer


# 🔧 CONFIG: yahan apne audio files ka folder set karo
# Agar files root me hi hain (jaise generated_music.wav, ...), to "." rehne do
AUDIO_DIR = "outputs_music_generator"          # e.g. ".", "outputs", "generated_audio", etc.
MAX_SAMPLES = 20         # mentor ne 20 bola hai


def find_audio_files():
    """MP3 / WAV files dhoondo given folder me."""
    patterns = [
        os.path.join(AUDIO_DIR, "*.mp3"),
        # os.path.join(AUDIO_DIR, "*.wav"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    # thoda sort kar lete hain (latest last)
    files = sorted(files, key=os.path.getmtime)
    if len(files) > MAX_SAMPLES:
        files = files[-MAX_SAMPLES:]  # last 20 (ya jitna set hai)
    return files


def main():
    scorer = QualityScorer()
    files = find_audio_files()

    if not files:
        print("No audio files found. Check AUDIO_DIR path.")
        return

    print(f"Found {len(files)} audio files. Scoring them now...\n")

    reports = []
    overall_scores = []

    for idx, fpath in enumerate(files, start=1):
        print(f"[{idx}] {fpath}")
        # yahan expected duration approx daal sakte ho.
        # Agar tum normally 10 sec request karte ho, to duration=10:
        expected_params = {
            "duration": 10,
            "mood": "calm",  # generic; zarurat ho to change
        }

        rep = scorer.score_audio(audio_file=fpath, expected_params=expected_params)

        overall = rep.get("overall_score", 0.0)
        scores = rep.get("scores", {})
        duration_sec = rep.get("duration_sec", None)

        print(f"   Overall score: {overall:.2f}")
        print(f"   Duration:      {duration_sec:.2f}s" if duration_sec is not None else "   Duration: N/A")
        print(f"   Metrics:       {scores}\n")

        overall_scores.append(overall)
        reports.append(
            {
                "index": idx,
                "file": os.path.abspath(fpath),
                "overall_score": overall,
                "duration_sec": duration_sec,
                "scores": scores,
            }
        )

    # Aggregate stats
    summary = {
        "num_samples": len(reports),
        "average_overall_score": mean(overall_scores) if overall_scores else 0.0,
        "min_score": min(overall_scores) if overall_scores else 0.0,
        "max_score": max(overall_scores) if overall_scores else 0.0,
        "below_threshold_count": sum(1 for s in overall_scores if s < 65.0),
        "threshold": 65.0,
    }

    result = {
        "summary": summary,
        "samples": reports,
    }

    out_path = "quality_report_20.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("----------------------------------------------------")
    print("Done. Report saved to:", out_path)
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
