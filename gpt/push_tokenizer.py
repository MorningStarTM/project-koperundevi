import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def main():
    parser = argparse.ArgumentParser(description="Push a trained tokenizer.json to Hugging Face Hub.")
    parser.add_argument("--tokenizer", required=True, help="Path to the local tokenizer JSON file.")
    parser.add_argument("--hf-repo-id", default="ErnestBeckham/gpt-v1")
    parser.add_argument("--hf-token", default=None, help="Falls back to HF_TOKEN in .env / env var.")
    parser.add_argument("--hf-private", action="store_true", default=True)
    parser.add_argument("--path-in-repo", default="tokenizer.json")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("No HF token found. Pass --hf-token or set HF_TOKEN in .env.")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.hf_repo_id, private=args.hf_private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=args.tokenizer,
        path_in_repo=args.path_in_repo,
        repo_id=args.hf_repo_id,
        commit_message="Add/update BPE tokenizer",
    )
    print(f"Pushed {args.tokenizer} to https://huggingface.co/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
