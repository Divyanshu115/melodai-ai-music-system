import os
import time

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile

# ----------------- CONFIG -----------------

MODEL_ID = "facebook/musicgen-small"

# 8 prompts from the task
PROMPTS = [
    "upbeat happy pop music",
    "sad slow piano melody",
    "energetic electronic dance music",
    "calm peaceful ambient sounds",
    "romantic acoustic guitar",
    "intense dramatic orchestral",
    "groovy funk bass",
    "mysterious dark atmospheric",
]

OUTPUT_DIR = "outputs_task1_2"

# Generation hyperparameters (you can tweak these)
DURATION_SECONDS = 4          # approx length of clip in seconds (keep small for your RAM)
TEMPERATURE = 1.0             # randomness / creativity
TOP_K = 250                   # top-k sampling
TOP_P = 0.95                  # top-p (nucleus) sampling
CFG_COEF = 3.0                # classifier-free guidance strength (guidance_scale)

# MusicGen uses about ~50 tokens per second at 32kHz EnCodec
TOKENS_PER_SECOND = 50

# ------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using GPU")
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA not available – using CPU (slower)")

    print(f"Loading model: {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

    # Optional: bfloat16 can sometimes save memory, but CPU pe dikkat ho sakti hai
    # isliye simple float32 hi rakhte hain
    model.to(device)

    # Make sure pad/eos tokens set (safe side)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = processor.tokenizer.eos_token_id

    max_new_tokens = int(DURATION_SECONDS * TOKENS_PER_SECOND)
    print(f"Approx duration per clip: {DURATION_SECONDS}s  (~{max_new_tokens} tokens)")
    print("Starting generation for 8 prompts...\n")

    results = []

    for idx, prompt in enumerate(PROMPTS, start=1):
        print(f"=== [{idx}/8] Prompt: {prompt} ===")

        # 1) Prepare inputs
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)

        # 2) Generate audio
        start_time = time.time()
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                guidance_scale=CFG_COEF,   # same idea as cfg_coef in docs
                max_new_tokens=max_new_tokens,
            )
        gen_time = time.time() - start_time

        # 3) Extract numpy audio and save as .wav
        # Shape: (batch_size, num_channels, sequence_length)
        audio_tensor = audio_values[0, 0].cpu().numpy()
        sampling_rate = model.config.audio_encoder.sampling_rate

        filename = os.path.join(
            OUTPUT_DIR,
            f"musicgen_{idx:02d}.wav"
        )
        wavfile.write(filename, rate=sampling_rate, data=audio_tensor)

        print(f"   -> Saved: {filename}")
        print(f"   -> Generation time: {gen_time:.2f} seconds\n")

        results.append((prompt, filename, gen_time))

    print("===== SUMMARY =====")
    for i, (prompt, fname, t) in enumerate(results, start=1):
        print(f"[{i}] {os.path.basename(fname)}")
        print(f"    Prompt: {prompt}")
        print(f"    Time: {t:.2f} s")
    print("\nAll 8 clips generated in folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
