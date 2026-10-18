import argparse
import os
import random

from bpe import BPETokenizer


def read_sample(path, sample_bytes, seed):
    """Read either the whole file or a random contiguous slice of it.

    Training scans the sample on every merge, so for a large corpus pass
    --sample-bytes (e.g. 200_000_000) instead of training on the full file.
    """
    if sample_bytes <= 0:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    file_size = 0
    with open(path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()

    sample_bytes = min(sample_bytes, file_size)
    random.seed(seed)
    start = random.randint(0, max(file_size - sample_bytes, 0))

    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(sample_bytes)

    # Drop partial utf-8 sequences at the edges of the slice.
    return raw.decode("utf-8", errors="ignore")


def push_to_hub(path, repo_id, token, private):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo="tokenizer.json",
        repo_id=repo_id,
        commit_message="Add/update BPE tokenizer",
    )
    print(f"Pushed tokenizer to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--data", required=True, help="Path to the training text file.")
    parser.add_argument("--output", required=True, help="Path to save the tokenizer JSON to.")
    parser.add_argument("--vocab-size", type=int, default=1800)
    parser.add_argument("--sample-bytes", type=int, default=200_000_000,
                         help="Bytes to sample from --data for training. 0 = use the whole file.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo-id", default="ErnestBeckham/gpt-v1")
    parser.add_argument("--hf-token", default=None, help="Falls back to the HF_TOKEN env var if omitted.")
    parser.add_argument("--hf-private", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Reading sample from {args.data} ...")
    text = read_sample(args.data, args.sample_bytes, args.seed)
    print(f"Training on {len(text):,} characters.")

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=args.vocab_size, verbose=args.verbose)

    tokenizer.save(args.output)
    print(f"Saved tokenizer (vocab_size={tokenizer.vocab_size}) to {args.output}")

    probe = text[:10000]
    ids = tokenizer.encode(probe)
    print(f"Sanity check: {len(probe)} chars -> {len(ids)} tokens "
          f"({len(probe) / len(ids):.2f} chars/token), "
          f"round-trip ok: {tokenizer.decode(ids) == probe}")

    if args.push_to_hub:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("--push-to-hub requires --hf-token or an HF_TOKEN env var.")
        push_to_hub(args.output, args.hf_repo_id, token, args.hf_private)


if __name__ == "__main__":
    main()
