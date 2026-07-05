"""テキスト生成: greedy, temperature, top-k, top-p サンプリング"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50,
             temperature=1.0, top_k=None, top_p=None, device="cpu"):
    model.eval()
    model.to(device)

    token_ids = tokenizer.encode(prompt)
    token_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # モデルが学習した最大長を超えないよう、直近のトークンのみを使用
        context = token_ids[:, -model.max_seq_len:]

        logits, _ = model(context)
        last_logits = logits[:, -1, :]  # 最後の位置のロジット: (batch, vocab_size)

        # temperature の適用
        last_logits = last_logits / max(temperature, 1e-5)

        # top-k フィルタリング: 上位 k 個以外は確率を -inf に設定
        if top_k is not None:
            topk_values, _ = torch.topk(last_logits, top_k)
            threshold = topk_values[:, -1].unsqueeze(-1)
            last_logits = torch.where(
                last_logits < threshold,
                torch.full_like(last_logits, float("-inf")),
                last_logits,
            )

        # top-p (nucleus) フィルタリング
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(last_logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # 累積確率が top_p を超える地点から除去対象としてマーク
            remove_mask = cumulative_probs > top_p
            # 最初のトークンは常に残し、候補がなくなるのを防ぐ
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False

            sorted_logits[remove_mask] = float("-inf")
            last_logits = torch.full_like(last_logits, float("-inf"))
            last_logits.scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # 確率的サンプリング

        token_ids = torch.cat([token_ids, next_token], dim=1)

    generated_ids = token_ids[0].tolist()
    return tokenizer.decode(generated_ids)
