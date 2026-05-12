from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import soundfile as sf

print("Loading model...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

prompt = ["peaceful relaxing lofi melody with soft piano and rain"]

inputs = processor(text=prompt, padding=True, return_tensors="pt")
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

sf.write("generated_music.wav", audio_values[0, 0].cpu().numpy(), model.config.audio_encoder.sampling_rate)

print("🎶 Music Generated → saved as generated_music.wav")
