"""Deploy RecallTrace to HuggingFace Spaces.

Usage:
    Set HF_TOKEN environment variable, then run:
    python deploy_to_hf.py
"""
import os
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "shreyabj18/RecallTrace"

if not TOKEN:
    print("ERROR: Set HF_TOKEN environment variable first.")
    print("  export HF_TOKEN=hf_your_token_here")
    exit(1)

api = HfApi(token=TOKEN)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"Uploading from: {PROJECT_DIR}")
print(f"Target: https://huggingface.co/spaces/{REPO_ID}")

api.upload_folder(
    folder_path=PROJECT_DIR,
    repo_id=REPO_ID,
    repo_type="space",
    token=TOKEN,
    ignore_patterns=[
        ".git/*", "__pycache__/*", ".pytest_cache/*",
        "deploy_to_hf.py", "*.pyc", ".mypy_cache/*",
    ],
)

print(f"\nDEPLOYMENT COMPLETE!")
print(f"Live at: https://huggingface.co/spaces/{REPO_ID}")
