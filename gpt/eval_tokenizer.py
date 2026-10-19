import argparse
import os

from bpe import BPETokenizer, SPLIT_PATTERN


def compression_ratio(tokenizer, text):
    ids = tokenizer.encode(text)
    return len(text) / len(ids), len(ids)


def fertility(tokenizer, text):
    """Average tokens per whitespace-delimited word."""
    words = text.split()
    if not words:
        return 0.0
    total_tokens = sum(len(tokenizer.encode(w)) for w in words)
    return total_tokens / len(words)


def vocab_utilization(tokenizer, text):
    ids = tokenizer.encode(text)
    used = len(set(ids))
    return used, tokenizer.vocab_size, used / tokenizer.vocab_size


def round_trip_ok(tokenizer, text):
    return tokenizer.decode(tokenizer.encode(text)) == text


def resolve_tokenizer_path(args):
    """Return a local path to tokenizer.json, downloading from HF Hub if requested."""
    if args.hf_repo_id:
        from huggingface_hub import hf_hub_download

        token = args.hf_token or os.environ.get("HF_TOKEN")
        return hf_hub_download(repo_id=args.hf_repo_id, filename=args.hf_filename, token=token)

    if not args.tokenizer:
        raise SystemExit("Pass either --tokenizer <local path> or --hf-repo-id <repo>.")
    return args.tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained BPE tokenizer on held-out text.")
    parser.add_argument("--tokenizer", default=None, help="Path to a local tokenizer.json")
    parser.add_argument("--hf-repo-id", default=None,
                         help="Download tokenizer from this HF Hub repo instead, e.g. ErnestBeckham/gpt-v1")
    parser.add_argument("--hf-filename", default="tokenizer.json", help="Filename within the HF repo.")
    parser.add_argument("--hf-token", default=None, help="Falls back to the HF_TOKEN env var if omitted.")
    parser.add_argument("--eval-data", required=True,
                         help="Path to held-out text NOT used in training (e.g. literature_2.txt).")
    parser.add_argument("--max-chars", type=int, default=2_000_000,
                         help="Cap how much of --eval-data to read, for speed.")
    args = parser.parse_args()

    tokenizer_path = resolve_tokenizer_path(args)

    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    with open(args.eval_data, "r", encoding="utf-8") as f:
        text = f.read(args.max_chars) if args.max_chars > 0 else f.read()

    print(f"Tokenizer: {tokenizer_path} (vocab_size={tokenizer.vocab_size})")
    print(f"Eval data: {args.eval_data} ({len(text):,} chars)")
    print()

    ratio, num_tokens = compression_ratio(tokenizer, text)
    print(f"Compression ratio: {ratio:.3f} chars/token  ({num_tokens:,} tokens)")

    fert = fertility(tokenizer, text)
    print(f"Fertility: {fert:.3f} tokens/word")

    used, total, pct = vocab_utilization(tokenizer, text)
    print(f"Vocab utilization: {used:,}/{total:,} tokens used ({pct:.1%})")

    probe = text[:10000]
    print(f"Round-trip exact on first 10k chars: {round_trip_ok(tokenizer, probe)}")


if __name__ == "__main__":
    main()
