import json
import os
import subprocess
from collections import Counter
from datetime import datetime

import requests
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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

CMD_TYPE_NAMES = [
    "filesystem", "git", "process", "network", "package", "docker",
    "k8s", "python", "node", "system", "text_processing", "ssh", "misc",
]


def build_cmd_type_class_weights(dataset):
    target_types = [
        window_target["cmd_type"]
        for _, window_target in dataset._get_windows()
    ]
    counts = Counter(target_types)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(CMD_TYPE_NAMES) * counts.get(name, 1)) for name in CMD_TYPE_NAMES],
        dtype=torch.float,
        device=DEVICE,
    )
    return weights / weights.sum() * len(CMD_TYPE_NAMES)


class_weights = build_cmd_type_class_weights(train_dataset)
criterion = CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
cmd_type_criterion = CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

warmup_steps = max(1, int(0.1 * EPOCHS * len(train_loader)))
total_training_steps = EPOCHS * len(train_loader)
warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
cosine = CosineAnnealingLR(
    optimizer,
    T_max=max(1, total_training_steps - warmup_steps),
    eta_min=1e-5,
)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
ALPHA = 0.3

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
            decoder_input = target_ids[:, :-1]
            decoder_target = target_ids[:, 1:]
            logits, _ = model(input_ids, context, decoder_input)
            loss = criterion(logits.transpose(1, 2), decoder_target)
            total_loss += loss.item()
            total_steps += 1
    return total_loss / total_steps


def evaluate_cmd_type_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input"].to(device)
            context = batch["context"].to(device)
            target_ids = batch["target"].to(device)
            target_cmd_types = batch["target_cmd_type"].to(device)
            _, cmd_type_logits = model(input_ids, context, target_ids[:, :-1])
            preds = cmd_type_logits.argmax(dim=-1)
            correct += (preds == target_cmd_types).sum().item()
            total += target_cmd_types.size(0)
    return correct / total if total > 0 else 0.0

def notify(title, message, level="info", event="completed", best_val_loss=0, step=0):
    token = os.environ.get("BEACON_TOKEN", "")
    if not token:
        print("[beacon] no token set, skipping")
        return
    try:
        r = requests.post(
            "https://beacon.sijibomi.com/emit",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "title": title,
                "message": message,
                "source": "mnesh-training",
                "event": event,
                "level": level,
                "channel": "email",
                "metadata": {
                    "best_val_loss": best_val_loss,
                    "total_steps": step,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                }
            }
        )
        print(f"[beacon] {r.status_code}")
    except Exception as e:
        print(f"[beacon] failed: {e}")

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
        target_cmd_types = batch["target_cmd_type"].to(DEVICE)

        optimizer.zero_grad()

        decoder_input  = target_ids[:, :-1]   # drop last token
        decoder_target = target_ids[:, 1:]    # drop first token (<s>)

        logits, cmd_type_logits = model(input_ids, context, decoder_input)
        cmd_loss = criterion(logits.transpose(1, 2), decoder_target)
        type_loss = cmd_type_criterion(cmd_type_logits, target_cmd_types)
        loss = cmd_loss + ALPHA * type_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"epoch {epoch+1} | step {step} | loss {loss.item():.4f} "
                f"| cmd {cmd_loss.item():.4f} | type {type_loss.item():.4f} | lr {lr:.6f}"
            )

        if step % EVAL_EVERY == 0 and step > 0:
            val_loss = evaluate(model, val_loader, criterion, DEVICE)
            type_acc = evaluate_cmd_type_accuracy(model, val_loader, DEVICE)
            print(
                f"epoch {epoch+1} | step {step} | train {loss.item():.4f} "
                f"| val {val_loss:.4f} | type_acc {type_acc:.4f}"
            )

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, step, val_loss, "checkpoints/mnesh_best.pt")
                print(f"new best model saved — val loss {val_loss:.4f}")

            model.train()

        step += 1

    # save end of epoch checkpoint
    save_checkpoint(model, optimizer, epoch, step, loss.item(), f"checkpoints/mnesh_epoch_{epoch+1}.pt")
    print(f"epoch {epoch+1} complete")
    notify(
        title=f"mnesh epoch {epoch+1} complete",
        message=f"epoch {epoch+1}/{EPOCHS} finished at step {step} with train loss {loss.item():.4f}",
        level="info",
        event="epoch_completed",
        best_val_loss=best_val_loss,
        step=step
    )

print("training complete")

notify(
    title="mnesh training complete",
    message=f"4 epochs done. best val loss: {best_val_loss:.4f} at step {step}",
    level="info",
    event="completed",
    best_val_loss=best_val_loss,
    step=step
)

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
