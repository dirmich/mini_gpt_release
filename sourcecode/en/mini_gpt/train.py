"""Training Loop"""
import torch


def train_model(model, dataloader, num_epochs, learning_rate, device):
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Gradually reduce the learning rate towards the end of training to ensure more stable convergence
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
            # Prevent gradient explosion: limit the gradient magnitude within a certain range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1
            history.append(loss.item())

            if step % 50 == 0:
                print(f"epoch {epoch+1} | step {step} | loss {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"=== epoch {epoch+1} average loss: {avg_loss:.4f} ===")

    return history
