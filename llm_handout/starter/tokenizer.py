"""Byte-level BPE tokenizer, trained from scratch (pure Python stdlib) on
train_corpus.txt only. Falls back to raw bytes for anything not covered by a
merge, so it stays LOSSLESS on arbitrary UTF-8 exactly like the baseline
ByteTokenizer:

  1. Pre-tokenize text into script/whitespace/punctuation runs (Unicode
     category based, not just ASCII \\w) so Devanagari consonant+matra
     clusters aren't split apart the way a plain \\w regex would split them.
  2. Each run is UTF-8-encoded to bytes (base vocab = 256 raw byte values —
     this is what guarantees the byte fallback / losslessness).
  3. Greedily apply learned BPE merges within each run, in trained order.

decode() just concatenates each token's fixed byte string and UTF-8-decodes
the result — since every token id always maps to the same byte string and
merges never destroy bytes, concatenation always reconstructs the exact
original bytes, so decode(encode(text)) == text unconditionally.

Merges are learned once (see train_and_save at the bottom / `python
tokenizer.py`) and cached in `bpe_merges.json` next to this file, loaded with
no internet access, resolved relative to __file__.
"""
import json
import unicodedata
import collections
import heapq
from pathlib import Path

ASSET_NAME = "bpe_merges.json"
VOCAB_SIZE = 2048


def _char_class(c):
    if c.isspace():
        return "space"
    cat = unicodedata.category(c)
    if cat[0] in ("L", "M", "N") or c == "_":
        return "word"
    return "other"


def _raw_chunks(text):
    chunks = []
    if not text:
        return chunks
    cur_class = _char_class(text[0])
    cur = [text[0]]
    for c in text[1:]:
        cls = _char_class(c)
        if cls == cur_class:
            cur.append(c)
        else:
            chunks.append((cur_class, "".join(cur)))
            cur_class, cur = cls, [c]
    chunks.append((cur_class, "".join(cur)))
    return chunks


def pretokenize(text):
    """Split text into chunks that concatenate back to `text` exactly.
    A single trailing space of a whitespace run is attached to the next
    word/other chunk (GPT-2 style) so common " word" pairs can merge."""
    chunks = _raw_chunks(text)
    out = []
    i, n = 0, len(chunks)
    while i < n:
        cls, chunk = chunks[i]
        if cls == "space" and i + 1 < n and chunks[i + 1][0] in ("word", "other"):
            if len(chunk) > 1:
                out.append(chunk[:-1])
                out.append(chunk[-1] + chunks[i + 1][1])
            else:
                out.append(chunk + chunks[i + 1][1])
            i += 2
        else:
            out.append(chunk)
            i += 1
    return out


def train_bpe(text, vocab_size):
    """Byte-pair-encode merge learning, restricted to pairs inside a single
    pretokenize() chunk. Uses word-frequency counting (not raw byte-stream
    scanning) with incremental pair-count updates, so it stays fast: only
    words containing the merged pair are touched per merge, not the whole
    corpus."""
    words = pretokenize(text)
    freq = collections.Counter(words)
    uniq_words = list(freq.keys())
    word_freq = [freq[w] for w in uniq_words]
    word_tokens = [list(w.encode("utf-8")) for w in uniq_words]

    pair_counts = collections.Counter()
    pair_to_words = collections.defaultdict(set)
    for wi, toks in enumerate(word_tokens):
        f = word_freq[wi]
        for a, b in zip(toks, toks[1:]):
            pair_counts[(a, b)] += f
            pair_to_words[(a, b)].add(wi)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges = []
    next_id = 256
    while next_id < vocab_size:
        pair = None
        while heap:
            negc, cand = heapq.heappop(heap)
            if pair_counts.get(cand, 0) == -negc and -negc > 0:
                pair = cand
                break
        if pair is None:
            break
        a, b = pair
        new_id = next_id
        merges.append(((a, b), new_id))

        for wi in list(pair_to_words[pair]):
            toks = word_tokens[wi]
            f = word_freq[wi]
            if len(toks) < 2:
                continue
            new_toks = []
            i = 0
            changed = False
            while i < len(toks):
                if i < len(toks) - 1 and toks[i] == a and toks[i + 1] == b:
                    if new_toks:
                        old_pair = (new_toks[-1], a)
                        pair_counts[old_pair] -= f
                        pair_to_words[old_pair].discard(wi)
                    if i + 2 < len(toks):
                        old_pair = (b, toks[i + 2])
                        pair_counts[old_pair] -= f
                        pair_to_words[old_pair].discard(wi)
                    new_toks.append(new_id)
                    if len(new_toks) >= 2:
                        np_ = (new_toks[-2], new_id)
                        pair_counts[np_] += f
                        pair_to_words[np_].add(wi)
                        heapq.heappush(heap, (-pair_counts[np_], np_))
                    changed = True
                    i += 2
                else:
                    new_toks.append(toks[i])
                    i += 1
            if changed:
                for k in range(len(new_toks) - 1):
                    if new_toks[k] == new_id or new_toks[k + 1] == new_id:
                        pair_to_words[(new_toks[k], new_toks[k + 1])].add(wi)
                word_tokens[wi] = new_toks
        pair_counts[pair] = 0
        pair_to_words[pair] = set()
        next_id += 1

    return merges


class BPETokenizer:
    def __init__(self, merges):
        self.merges = [((a, b), new_id) for (a, b), new_id in merges]
        self.vocab_size = 256 + len(self.merges)
        self._id_to_bytes = {i: bytes([i]) for i in range(256)}
        self._merge_rank = {}
        self._merge_id = {}
        for rank, ((a, b), new_id) in enumerate(self.merges):
            self._id_to_bytes[new_id] = self._id_to_bytes[a] + self._id_to_bytes[b]
            self._merge_rank[(a, b)] = rank
            self._merge_id[(a, b)] = new_id

    def _encode_chunk(self, tokens):
        tokens = list(tokens)
        while len(tokens) >= 2:
            best_rank, best_idx = None, None
            for i in range(len(tokens) - 1):
                r = self._merge_rank.get((tokens[i], tokens[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_idx = r, i
            if best_idx is None:
                break
            a, b = tokens[best_idx], tokens[best_idx + 1]
            new_id = self._merge_id[(a, b)]
            tokens = tokens[:best_idx] + [new_id] + tokens[best_idx + 2:]
        return tokens

    def encode(self, text):
        ids = []
        for chunk in pretokenize(text):
            ids.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
        return ids

    def decode(self, ids):
        return b"".join(self._id_to_bytes[i] for i in ids).decode("utf-8")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": [[list(p), nid] for p, nid in self.merges]}, f)


def _asset_path():
    return Path(__file__).resolve().with_name(ASSET_NAME)


def load(path=None):
    """Return the tokenizer used by train.py and evaluate.py. Loads the
    merges trained by train_and_save() from a JSON file saved next to this
    module; no internet, no args required."""
    asset = _asset_path() if path is None else Path(path)
    with open(asset, encoding="utf-8") as f:
        payload = json.load(f)
    merges = [(tuple(p), nid) for p, nid in payload["merges"]]
    return BPETokenizer(merges)


def train_and_save(corpus_path, vocab_size=VOCAB_SIZE, out_path=None):
    text = open(corpus_path, encoding="utf-8").read()
    merges = train_bpe(text, vocab_size)
    tok = BPETokenizer(merges)
    tok.save(out_path or _asset_path())
    return tok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/train_corpus.txt")
    ap.add_argument("--vocab_size", type=int, default=VOCAB_SIZE)
    args = ap.parse_args()
    tok = train_and_save(args.data, args.vocab_size)
    print(f"trained BPE tokenizer: vocab_size={tok.vocab_size}, "
          f"saved to {_asset_path()}")
