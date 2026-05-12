from huggingface_hub import login
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Read token from environment
token = os.getenv("HF_TOKEN")

# Login
login(token=token, add_to_git_credential=False)

print("✅ Logged in to Hugging Face successfully!")