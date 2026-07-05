"""学習ループ"""
import torch


def train_model(model, dataloader, num_epochs, learning_rate, device):
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # 学習の後半にかけて学習率を徐々に減少させ、より安定した収束を実現する
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(dataloader)
    )

    history = []
    step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits, loss = model(x_batch, targets=y_batch)

            optimizer.zero_grad()
            loss.backward()
            # 勾配爆発の防止: 勾配の大きさを一定の範囲内に制限
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1
            history.append(loss.item())

            if step % 50 == 0:
                print(f"epoch {epoch+1} | step {step} | loss {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"=== epoch {epoch+1} 平均 loss: {avg_loss:.4f} ===")

    return history
