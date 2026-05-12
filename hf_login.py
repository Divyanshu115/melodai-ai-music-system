from huggingface_hub import login

# ⭐ Paste your token inside the quotes:
import os

token = os.getenv("HF_TOKEN")

login(token=TOKEN, add_to_git_credential=False)
print("✅ Logged in to Hugging Face successfully!")
