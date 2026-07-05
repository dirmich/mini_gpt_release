"""
mini_gpt 全体パイプライン・デモスクリプト

本の第2章から第8章の内容を順番に実行します：
トークナイザーの学習 -> モデルの生成 -> 学習 -> テキスト生成

実行方法：
    pip install -r requirements.txt
    python run_demo.py
"""
import torch
from torch.utils.data import DataLoader

from mini_gpt.tokenizer import BPETokenizer
from mini_gpt.dataset import TextDataset
from mini_gpt.model import MiniGPT
from mini_gpt.train import train_model
from mini_gpt.generate import generate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用するデバイス:", device)

    # 1) 学習用テキストの準備 (任意のテキストに置き換え可能)
    try:
        from datasets import load_dataset
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2%]")
        text_corpus = "\n".join([t for t in raw_dataset["text"] if len(t.strip()) > 0])
    except Exception as e:
        print("datasets ライブラリによるダウンロードに失敗しました。内蔵サンプルテキストを使用します:", e)
        text_corpus = (
            "The Transformer architecture relies entirely on attention mechanisms. "
            "Language models predict the next token given previous tokens. "
            "Attention lets each token look at every other token in the sequence. "
        ) * 200

    print("コーパスの文字数:", len(text_corpus))

    # 2) トークナイザーの学習
    tokenizer = BPETokenizer()
    tokenizer.train(text_corpus, vocab_size=1000)
    all_token_ids = tokenizer.encode(text_corpus)
    print("総トークン数:", len(all_token_ids), "| 語彙サイズ:", len(tokenizer.vocab))

    # 3) データセット / データローダー
    seq_len = 64
    dataset = TextDataset(all_token_ids, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 4) モデルの生成
    model = MiniGPT(
        vocab_size=len(tokenizer.vocab),
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        max_seq_len=seq_len,
        dropout=0.1,
    )
    print("パラメータ数:", f"{model.num_parameters():,}")

    # 5) 学習
    train_model(model, dataloader, num_epochs=3, learning_rate=3e-4, device=device)

    # 6) 生成
    prompt = "The history of"
    print("\n=== 生成結果 (top-p) ===")
    print(generate(model, tokenizer, prompt, max_new_tokens=40,
                    temperature=0.8, top_p=0.9, device=device))

    # 7) チェックポイントの保存
    torch.save(model.state_dict(), "mini_gpt_checkpoint.pt")
    print("\nモデルを mini_gpt_checkpoint.pt として保存しました。")


if __name__ == "__main__":
    main()
