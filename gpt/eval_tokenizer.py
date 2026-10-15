import argparse

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained BPE tokenizer on held-out text.")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--eval-data", required=True,
                         help="Path to held-out text NOT used in training (e.g. literature_2.txt).")
    parser.add_argument("--max-chars", type=int, default=2_000_000,
                         help="Cap how much of --eval-data to read, for speed.")
    args = parser.parse_args()

    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)

    with open(args.eval_data, "r", encoding="utf-8") as f:
        text = f.read(args.max_chars) if args.max_chars > 0 else f.read()

    print(f"Tokenizer: {args.tokenizer} (vocab_size={tokenizer.vocab_size})")
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
