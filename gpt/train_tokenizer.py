import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--data", required=True, help="Path to the training text file.")
    parser.add_argument("--output", required=True, help="Path to save the tokenizer JSON to.")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--sample-bytes", type=int, default=200_000_000,
                         help="Bytes to sample from --data for training. 0 = use the whole file.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--verbose", action="store_true")
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


if __name__ == "__main__":
    main()
