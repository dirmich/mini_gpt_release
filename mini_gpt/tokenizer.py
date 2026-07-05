"""BPE(Byte Pair Encoding) 토크나이저 - 처음부터 구현"""
from collections import Counter, defaultdict


class BPETokenizer:
    def __init__(self):
        self.merges = {}          # (토큰1, 토큰2) -> 합쳐진 새 토큰 ID
        self.vocab = {}           # 토큰 ID -> 바이트 시퀀스
        self.token_to_id = {}     # 문자열 -> 토큰 ID

    def _get_pair_counts(self, token_ids_list):
        """모든 시퀀스에서 인접한 토큰 쌍의 등장 횟수를 센다."""
        counts = Counter()
        for ids in token_ids_list:
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] += 1
        return counts

    def _merge(self, ids, pair, new_id):
        """ids 안에서 pair를 new_id로 모두 합친다."""
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
        """텍스트로부터 BPE 어휘 사전을 학습한다."""
        # 1단계: 초기 어휘는 UTF-8 바이트 256개
        for i in range(256):
            self.vocab[i] = bytes([i])

        # 텍스트를 바이트 단위 정수 리스트로 변환
        ids = list(text.encode("utf-8"))
        token_ids_list = [ids]  # 문서가 하나뿐이라 리스트 안에 하나만

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
        """텍스트를 토큰 ID 리스트로 변환한다."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pair_counts = self._get_pair_counts([ids])
            # 학습 때 만들어진 병합 중, 등장한 쌍 가운데 가장 먼저 만들어진 병합을 우선 적용
            candidates = [p for p in pair_counts if p in self.merges]
            if not candidates:
                break
            best_pair = min(candidates, key=lambda p: self.merges[p])
            ids = self._merge(ids, best_pair, self.merges[best_pair])
        return ids

    def decode(self, ids):
        """토큰 ID 리스트를 다시 텍스트로 복원한다."""
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")
