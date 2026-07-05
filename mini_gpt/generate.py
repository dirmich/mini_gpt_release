"""텍스트 생성: greedy, temperature, top-k, top-p 샘플링"""
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
        # 모델이 학습할 때 본 최대 길이를 넘지 않도록 최근 토큰만 사용
        context = token_ids[:, -model.max_seq_len:]

        logits, _ = model(context)
        last_logits = logits[:, -1, :]  # 마지막 위치의 로짓: (batch, vocab_size)

        # temperature 적용
        last_logits = last_logits / max(temperature, 1e-5)

        # top-k 필터링: 상위 k개 외에는 확률을 -inf로
        if top_k is not None:
            topk_values, _ = torch.topk(last_logits, top_k)
            threshold = topk_values[:, -1].unsqueeze(-1)
            last_logits = torch.where(
                last_logits < threshold,
                torch.full_like(last_logits, float("-inf")),
                last_logits,
            )

        # top-p(nucleus) 필터링
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(last_logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # 누적 확률이 top_p를 넘어서는 지점부터는 제거 대상으로 표시
            remove_mask = cumulative_probs > top_p
            # 첫 번째 토큰은 항상 남겨서 후보가 아예 없는 경우를 방지
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False

            sorted_logits[remove_mask] = float("-inf")
            last_logits = torch.full_like(last_logits, float("-inf"))
            last_logits.scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # 확률적 샘플링

        token_ids = torch.cat([token_ids, next_token], dim=1)

    generated_ids = token_ids[0].tolist()
    return tokenizer.decode(generated_ids)
