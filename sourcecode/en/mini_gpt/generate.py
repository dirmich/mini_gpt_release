"""Text Generation: greedy, temperature, top-k, top-p sampling"""
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
        # Use only the most recent tokens to ensure they do not exceed the maximum length seen during model training
        context = token_ids[:, -model.max_seq_len:]

        logits, _ = model(context)
        last_logits = logits[:, -1, :]  # Logits at the last position: (batch, vocab_size)

        # Apply temperature
        last_logits = last_logits / max(temperature, 1e-5)

        # top-k filtering: set probabilities to -inf for everything except the top k
        if top_k is not None:
            topk_values, _ = torch.topk(last_logits, top_k)
            threshold = topk_values[:, -1].unsqueeze(-1)
            last_logits = torch.where(
                last_logits < threshold,
                torch.full_like(last_logits, float("-inf")),
                last_logits,
            )

        # top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(last_logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Mark as candidates for removal from the point where cumulative probability exceeds top_p
            remove_mask = cumulative_probs > top_p
            # Always keep the first token to prevent cases with no candidates
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False

            sorted_logits[remove_mask] = float("-inf")
            last_logits = torch.full_like(last_logits, float("-inf"))
            last_logits.scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # Stochastic sampling

        token_ids = torch.cat([token_ids, next_token], dim=1)

    generated_ids = token_ids[0].tolist()
    return tokenizer.decode(generated_ids)
