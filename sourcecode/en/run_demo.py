"""
mini_gpt full pipeline demo script

Executes the contents of Chapters 2 to 8 in order:
Tokenizer training -> Model creation -> Training -> Text generation

How to run:
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
    print("Device to use:", device)

    # 1) Prepare text for training (can be replaced with desired text)
    try:
        from datasets import load_dataset
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2%]")
        text_corpus = "\n".join([t for t in raw_dataset["text"] if len(t.strip()) > 0])
    except Exception as e:
        print("Failed to download with datasets library, using built-in sample text:", e)
        text_corpus = (
            "The Transformer architecture relies entirely on attention mechanisms. "
            "Language models predict the next token given previous tokens. "
            "Attention lets each token look at every other token in the sequence. "
        ) * 200

    print("Corpus character count:", len(text_corpus))

    # 2) Tokenizer training
    tokenizer = BPETokenizer()
    tokenizer.train(text_corpus, vocab_size=1000)
    all_token_ids = tokenizer.encode(text_corpus)
    print("Total token count:", len(all_token_ids), "| Vocab size:", len(tokenizer.vocab))

    # 3) Dataset / DataLoader
    seq_len = 64
    dataset = TextDataset(all_token_ids, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 4) Model creation
    model = MiniGPT(
        vocab_size=len(tokenizer.vocab),
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        max_seq_len=seq_len,
        dropout=0.1,
    )
    print("Number of parameters:", f"{model.num_parameters():,}")

    # 5) Training
    train_model(model, dataloader, num_epochs=3, learning_rate=3e-4, device=device)

    # 6) Generation
    prompt = "The history of"
    print("\n=== Generation Results (top-p) ===")
    print(generate(model, tokenizer, prompt, max_new_tokens=40,
                    temperature=0.8, top_p=0.9, device=device))

    # 7) Save checkpoint
    torch.save(model.state_dict(), "mini_gpt_checkpoint.pt")
    print("\nModel saved as mini_gpt_checkpoint.pt")


if __name__ == "__main__":
    main()
