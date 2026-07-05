"""BPE (Byte Pair Encoding) Tokenizer - Implementation from Scratch"""
from collections import Counter, defaultdict


class BPETokenizer:
    def __init__(self):
        self.merges = {}          # (token1, token2) -> merged new token ID
        self.vocab = {}           # token ID -> byte sequence
        self.token_to_id = {}     # string -> token ID

    def _get_pair_counts(self, token_ids_list):
        """Counts the frequency of adjacent token pairs in all sequences."""
        counts = Counter()
        for ids in token_ids_list:
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] += 1
        return counts

    def _merge(self, ids, pair, new_id):
        """Merges all pairs within ids into new_id."""
        merged = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    def train(self, text, vocab_size=500):
        """Trains a BPE vocabulary from text."""
        # Step 1: Initial vocabulary consists of 256 UTF-8 bytes
        for i in range(256):
            self.vocab[i] = bytes([i])

        # Convert text into a list of byte integers
        ids = list(text.encode("utf-8"))
        token_ids_list = [ids]  # Only one document, so just one item in the list

        num_merges = vocab_size - 256
        for step in range(num_merges):
            pair_counts = self._get_pair_counts(token_ids_list)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + step
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges[best_pair] = new_id
            token_ids_list = [self._merge(ids, best_pair, new_id) for ids in token_ids_list]

        return self

    def encode(self, text):
        """Converts text into a list of token IDs."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pair_counts = self._get_pair_counts([ids])
            # Among merges created during training, apply the earliest-created pair first
            candidates = [p for p in pair_counts if p in self.merges]
            if not candidates:
                break
            best_pair = min(candidates, key=lambda p: self.merges[p])
            ids = self._merge(ids, best_pair, self.merges[best_pair])
        return ids

    def decode(self, ids):
        """Restores a list of token IDs back to text."""
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")
