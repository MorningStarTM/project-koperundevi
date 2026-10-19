import json
import os
import regex as re
from tqdm import tqdm


# GPT-2 style text splitting pattern so merges don't cross word/whitespace/number
# boundaries. Works byte-level, so it is script-agnostic (fine for Tamil text).
SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    """Byte-level BPE tokenizer, trained from scratch (GPT-2 style)."""

    def __init__(self):
        self.merges = {}       # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)}  # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = SPLIT_PATTERN.findall(text)
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        progress = tqdm(range(num_merges), desc="Training BPE", unit="merge")
        for i in progress:
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)

            if not stats:
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            progress.set_postfix(pair=pair, occurrences=stats[pair])

            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} "
                      f"({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        text_chunks = SPLIT_PATTERN.findall(text)
        ids = []
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                stats = get_stats(chunk_ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                chunk_ids = merge(chunk_ids, pair, self.merges[pair])
            ids.extend(chunk_ids)
        return ids

    def decode(self, ids):
        token_bytes = b"".join(self.vocab[idx] for idx in ids)
        return token_bytes.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.vocab)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "merges": [[list(pair), idx] for pair, idx in self.merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for pair, idx in data["merges"]:
            pair = tuple(pair)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab
