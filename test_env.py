print("----- LIBRARY IMPORT TEST STARTED -----")

try:
    import torch
    print("Torch Loaded:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
except Exception as e:
    print("Torch Error:", e)

try:
    import torchaudio
    print("Torchaudio Loaded")
except Exception as e:
    print("Torchaudio Error:", e)

try:
    import streamlit
    print("Streamlit Loaded")
except Exception as e:
    print("Streamlit Error:", e)

try:
    import transformers
    print("Transformers Loaded")
except Exception as e:
    print("Transformers Error:", e)

try:
    import music21
    print("music21 Loaded")
except Exception as e:
    print("music21 Error:", e)

try:
    import mido
    print("mido Loaded")
except Exception as e:
    print("mido Error:", e)

try:
    import pretty_midi
    print("Pretty MIDI Loaded")
except Exception as e:
    print("Pretty MIDI Error:", e)

try:
    import pydub
    print("Pydub Loaded")
except Exception as e:
    print("Pydub Error:", e)

try:
    import scipy
    print("Scipy Loaded")
except Exception as e:
    print("Scipy Error:", e)

try:
    import openai
    print("OpenAI Loaded")
except Exception as e:
    print("OpenAI Error:", e)

try:
    import audiocraft
    print("Audiocraft Loaded")
except Exception as e:
    print("Audiocraft Error:", e)

print("----- TEST FINISHED -----")
