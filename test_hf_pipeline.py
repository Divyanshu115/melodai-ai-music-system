# test_hf_pipeline.py
# Single-run test to verify HF ModelManager + main_service pipeline

import os
import json

from backend.main_service import generate_music_pipeline


def main():
    print("Running HF pipeline test...\n")

    meta, params, enhanced = generate_music_pipeline(
    user_input="Intimate nylon-guitar and soft string quartet for romantic scene",
    duration=5,                      # chhota duration fast hoga
    variations=1,
    debug=False,
    model_name="facebook/musicgen-small",   # 👈 ab small test karein
)


    print("---- META SUMMARY ----")
    print(
        json.dumps(
            {
                "id": meta.get("id"),
                "error": meta.get("error"),
                "files": meta.get("files"),
                "model_used": meta.get("model_used"),
            },
            indent=2,
        )
    )

    files = meta.get("files") or []
    if not files:
        print("\nNo files returned in meta['files'].")
        return

    first = files[0]
    path = first.get("wav_raw") or first.get("mp3") or first.get("file")

    print("\nAudio file path:", path)

    if not path:
        print("No path field in first file entry.")
        return

    path = os.path.abspath(path)
    print("Absolute path:", path)
    print("Exists:", os.path.exists(path))

    if os.path.exists(path):
        size = os.path.getsize(path)
        print("Size (bytes):", size)
    else:
        print("WARNING: File does not exist on disk.")


if __name__ == "__main__":
    main()
