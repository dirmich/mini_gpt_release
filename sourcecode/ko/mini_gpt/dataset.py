"""텍스트 -> (입력, 정답) 학습 배치를 만드는 데이터셋"""
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        """token_ids: 전체 텍스트를 토크나이징한 긴 정수 리스트"""
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        # seq_len+1 크기의 조각을 겹치지 않게 뽑을 수 있는 시작 위치의 개수
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.token_ids[idx: idx + self.seq_len + 1]
        x = chunk[:-1]   # 입력: 앞의 seq_len개
        y = chunk[1:]    # 정답: 한 칸씩 민 seq_len개
        return x, y
