import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.dataset import MneshDatasetV1
from model.main import CFG, MneshModel

print("Welcome to Mnesh training session...")

# hyperparameters
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
EPOCHS        = 4
EVAL_EVERY    = 500
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# setup
train_dataset = MneshDatasetV1(split="train")
val_dataset   = MneshDatasetV1(split="val")
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
model         = MneshModel(CFG).to(DEVICE)
criterion     = CrossEntropyLoss(ignore_index=0)
optimizer     = Adam(model.parameters(), lr=LEARNING_RATE)

total_params = sum(p.numel() for p in model.parameters())
print(f"model parameters: {total_params:,}")
print(f"training on:      {DEVICE}")
print(f"train batches:    {len(train_loader)}")
print(f"val batches:      {len(val_loader)}")

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_steps = 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids  = batch["input"].to(device)
            context    = batch["context"].to(device)
            target_ids = batch["target"].to(device)
            logits = model(input_ids, context, target_ids)
            loss = criterion(logits.transpose(1, 2), target_ids)
            total_loss += loss.item()
            total_steps += 1
    return total_loss / total_steps

def save_checkpoint(model, optimizer, epoch, step, loss, path):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "step":      step,
        "loss":      loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"checkpoint saved → {path}")

# training loop
step = 0
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        input_ids  = batch["input"].to(DEVICE)
        context    = batch["context"].to(DEVICE)
        target_ids = batch["target"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(input_ids, context, target_ids)
        loss = criterion(logits.transpose(1, 2), target_ids)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"epoch {epoch+1} | step {step} | loss {loss.item():.4f}")
        if step % EVAL_EVERY == 0:
            val_loss = evaluate(model, val_loader, criterion, DEVICE)
            print(f"epoch {epoch+1} | step {step} | train {loss.item():.4f} | val {val_loss:.4f}")
            save_checkpoint(model, optimizer, epoch, step, val_loss, f"checkpoints/mnesh_step_{step}.pt")
            model.train()
        step += 1

print("training complete")
