"""
mini_gpt 전체 파이프라인 데모 스크립트

책의 2~8장 내용을 순서대로 실행합니다:
토크나이저 학습 -> 모델 생성 -> 학습 -> 텍스트 생성

실행 방법:
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
    print("사용할 디바이스:", device)

    # 1) 학습용 텍스트 준비 (원하는 텍스트로 교체 가능)
    try:
        from datasets import load_dataset
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2%]")
        text_corpus = "\n".join([t for t in raw_dataset["text"] if len(t.strip()) > 0])
    except Exception as e:
        print("datasets 라이브러리로 다운로드 실패, 내장 샘플 텍스트를 사용합니다:", e)
        text_corpus = (
            "The Transformer architecture relies entirely on attention mechanisms. "
            "Language models predict the next token given previous tokens. "
            "Attention lets each token look at every other token in the sequence. "
        ) * 200

    print("코퍼스 글자 수:", len(text_corpus))

    # 2) 토크나이저 학습
    tokenizer = BPETokenizer()
    tokenizer.train(text_corpus, vocab_size=1000)
    all_token_ids = tokenizer.encode(text_corpus)
    print("전체 토큰 수:", len(all_token_ids), "| 어휘 크기:", len(tokenizer.vocab))

    # 3) 데이터셋 / 데이터로더
    seq_len = 64
    dataset = TextDataset(all_token_ids, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 4) 모델 생성
    model = MiniGPT(
        vocab_size=len(tokenizer.vocab),
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        max_seq_len=seq_len,
        dropout=0.1,
    )
    print("파라미터 수:", f"{model.num_parameters():,}")

    # 5) 학습
    train_model(model, dataloader, num_epochs=3, learning_rate=3e-4, device=device)

    # 6) 생성
    prompt = "The history of"
    print("\n=== 생성 결과 (top-p) ===")
    print(generate(model, tokenizer, prompt, max_new_tokens=40,
                    temperature=0.8, top_p=0.9, device=device))

    # 7) 체크포인트 저장
    torch.save(model.state_dict(), "mini_gpt_checkpoint.pt")
    print("\n모델을 mini_gpt_checkpoint.pt 로 저장했습니다.")


if __name__ == "__main__":
    main()
