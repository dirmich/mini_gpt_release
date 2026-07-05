"""テキスト -> (入力, 正解) 学習バッチを作成するデータセット"""
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        """token_ids: 全文をトークナイズした長い整数のリスト"""
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        # seq_len+1 サイズのチャンクを重なりなく抽出できる開始位置の数
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.token_ids[idx: idx + self.seq_len + 1]
        x = chunk[:-1]   # 入力: 前方の seq_len 個
        y = chunk[1:]    # 正解: 1つずらした seq_len 個
        return x, y
