"""BPE (Byte Pair Encoding) トークナイザー - 最初から実装"""
from collections import Counter, defaultdict


class BPETokenizer:
    def __init__(self):
        self.merges = {}          # (トークン1, トークン2) -> 結合された新しいトークン ID
        self.vocab = {}           # トークン ID -> バイトシーケンス
        self.token_to_id = {}     # 文字列 -> トークン ID

    def _get_pair_counts(self, token_ids_list):
        """すべてのシーケンスにおいて、隣接するトークンのペアの出現回数をカウントする。"""
        counts = Counter()
        for ids in token_ids_list:
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] += 1
        return counts

    def _merge(self, ids, pair, new_id):
        """ids 内のペアを new_id ですべて結合する。"""
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
        """テキストから BPE 語彙辞書を学習する。"""
        # ステップ 1: 初期語彙は UTF-8 バイト 256 個
        for i in range(256):
            self.vocab[i] = bytes([i])

        # テキストをバイト単位の整数リストに変換
        ids = list(text.encode("utf-8"))
        token_ids_list = [ids]  # ドキュメントが一つのみのため、リスト内に一つだけ保持

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
        """テキストをトークン ID リストに変換する。"""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pair_counts = self._get_pair_counts([ids])
            # 学習時に作成された結合のうち、出現したペアの中で最も早く作成されたものを優先的に適用
            candidates = [p for p in pair_counts if p in self.merges]
            if not candidates:
                break
            best_pair = min(candidates, key=lambda p: self.merges[p])
            ids = self._merge(ids, best_pair, self.merges[best_pair])
        return ids

    def decode(self, ids):
        """トークン ID リストを再びテキストに復元する。"""
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")
