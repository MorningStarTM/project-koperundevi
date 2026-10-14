import argparse
import time

import torch

import gpt_v1
from bpe import BPETokenizer
from trainer import Trainer, TrainerConfig


def read_text(path, max_chars):
    with open(path, "r", encoding="utf-8") as f:
        return f.read(max_chars) if max_chars > 0 else f.read()


def encode_with_progress(tokenizer, text, chunk_chars=5_000_000):
    """Encode text in fixed-size slices, printing progress.

    Slicing on raw character boundaries (rather than tokenizer chunk
    boundaries) means a merge opportunity at a slice edge can be missed;
    that's a negligible loss for training data and keeps memory/feedback
    manageable on very large corpora.
    """
    ids = []
    total = len(text)
    start_time = time.time()

    for start in range(0, total, chunk_chars):
        piece = text[start:start + chunk_chars]
        ids.extend(tokenizer.encode(piece))

        done = min(start + chunk_chars, total)
        elapsed = time.time() - start_time
        print(f"  encoded {done:,}/{total:,} chars ({done / total:.1%}), {elapsed:.1f}s elapsed", end="\r")

    print()
    return ids


def main():
    parser = argparse.ArgumentParser(description="Pretrain GPTLanguageModel with a trained BPE tokenizer.")
    parser.add_argument("--data", required=True, help="Path to the training text file.")
    parser.add_argument("--tokenizer", required=True, help="Path to a tokenizer JSON from train_tokenizer.py.")
    parser.add_argument("--output-dir", default="../models")
    parser.add_argument("--max-chars", type=int, default=0, help="Limit input text size for quick test runs. 0 = full file.")
    parser.add_argument("--val-fraction", type=float, default=0.1)

    # model hyperparameters (override gpt_v1 module globals before construction)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-emb", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # trainer hyperparameters
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iters", type=int, default=6000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")

    # Hugging Face Hub
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo-id", default="ErnestBeckham/gpt-v1")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-private", action="store_true", default=True)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    gpt_v1.device = device
    gpt_v1.block_size = args.block_size
    gpt_v1.n_emb = args.n_emb
    gpt_v1.n_layers = args.n_layers
    gpt_v1.n_head = args.n_head
    gpt_v1.dropout = args.dropout

    print(f"Loading tokenizer from {args.tokenizer} ...")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    print(f"Reading data from {args.data} ...")
    text = read_text(args.data, args.max_chars)
    print(f"Read {len(text):,} characters.")

    print("Encoding data ...")
    ids = encode_with_progress(tokenizer, text)
    data = torch.tensor(ids, dtype=torch.long)
    print(f"Encoded into {len(data):,} tokens.")

    n = int((1 - args.val_fraction) * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"train tokens: {len(train_data):,}, val tokens: {len(val_data):,}")

    model = gpt_v1.GPTLanguageModel(vocab_size=tokenizer.vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    config = TrainerConfig(
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.lr,
        grad_clip=args.grad_clip,
        device=device,
        checkpoint_dir=args.output_dir,
        resume=args.resume,
        push_to_hub=args.push_to_hub,
        hf_repo_id=args.hf_repo_id,
        hf_token=args.hf_token,
        hf_private=args.hf_private,
    )

    trainer = Trainer(model, train_data, val_data, config)
    trainer.train()


if __name__ == "__main__":
    main()
