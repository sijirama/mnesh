import json
import os
import subprocess
from datetime import datetime

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.dataset import MneshDatasetV1
from model.main import CFG, MneshModel

# ── intro ──────────────────────────────────────────────

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
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
model         = MneshModel(CFG).to(DEVICE)
criterion     = CrossEntropyLoss(ignore_index=0)
optimizer     = Adam(model.parameters(), lr=LEARNING_RATE)

total_params = sum(p.numel() for p in model.parameters())
print(f"model parameters: {total_params:,}")
print(f"training on:      {DEVICE}")
print(f"train batches:    {len(train_loader)}")
print(f"val batches:      {len(val_loader)}")

# ── helpers ──────────────────────────────────────────────

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

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"], checkpoint["step"], checkpoint["loss"]

# ── training loop ──────────────────────────────────────────────

step = 0
loss = torch.tensor(0.0)
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:

        input_ids  = batch["input"].to(DEVICE)
        context    = batch["context"].to(DEVICE)
        target_ids = batch["target"].to(DEVICE)

        optimizer.zero_grad()

        decoder_input  = target_ids[:, :-1]   # drop last token
        decoder_target = target_ids[:, 1:]    # drop first token (<s>)

        logits = model(input_ids, context, target_ids)
        loss   = criterion(logits.transpose(1, 2), target_ids)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f"epoch {epoch+1} | step {step} | loss {loss.item():.4f}")

        if step % EVAL_EVERY == 0 and step > 0:
            val_loss = evaluate(model, val_loader, criterion, DEVICE)
            print(f"epoch {epoch+1} | step {step} | train {loss.item():.4f} | val {val_loss:.4f}")

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, step, val_loss, "checkpoints/mnesh_best.pt")
                print(f"new best model saved — val loss {val_loss:.4f}")

            # remove the step checkpoint save - taking too much space
            # save_checkpoint(model, optimizer, epoch, step, val_loss, f"checkpoints/mnesh_step_{step}.pt")
            model.train()

        step += 1

    # save end of epoch checkpoint
    save_checkpoint(model, optimizer, epoch, step, loss.item(), f"checkpoints/mnesh_epoch_{epoch+1}.pt")
    print(f"epoch {epoch+1} complete")

print("training complete")

# generate run report
report = {
    "timestamp":       datetime.now().isoformat(),
    "epochs":          EPOCHS,
    "batch_size":      BATCH_SIZE,
    "learning_rate":   LEARNING_RATE,
    "total_steps":     step,
    "best_val_loss":   best_val_loss,
    "model_params":    total_params,
    "device":          DEVICE,
    "cfg":             CFG,
    "final_train_loss": loss.item(),
    "checkpoints_saved": os.listdir("checkpoints") if os.path.exists("checkpoints") else [],
}

os.makedirs("runs", exist_ok=True)
report_path = f"runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print("\n── run report ──────────────────────────────")
print(f"timestamp:        {report['timestamp']}")
print(f"total steps:      {report['total_steps']:,}")
print(f"best val loss:    {report['best_val_loss']:.4f}")
print(f"final train loss: {report['final_train_loss']:.4f}")
print(f"report saved →    {report_path}")
print("────────────────────────────────────────────\n")

# auto stop pod when done
pod_id = os.environ.get("RUNPOD_POD_ID", "")
if pod_id:
    print(f"stopping pod {pod_id}...")
    subprocess.run(["runpodctl", "stop", "pod", pod_id])
