import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

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
MODEL_VERSION = Path("model/VERSION").read_text().strip()

# setup
train_dataset = MneshDatasetV1(split="train")
val_dataset   = MneshDatasetV1(split="val")
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
model         = MneshModel(CFG).to(DEVICE)

CMD_TYPE_NAMES = [
    "filesystem", "git", "process", "network", "package", "container",
    "python", "node", "system", "text_processing", "misc",
]


def build_cmd_type_class_weights(dataset):
    target_types = [
        dataset.normalize_cmd_type(window_target["cmd_type"])
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
criterion = CrossEntropyLoss(ignore_index=0, label_smoothing=0.1, reduction="none")
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


def get_alpha(epoch):
    return 1.0 if epoch < 2 else 0.3


def weighted_sequence_loss(logits, decoder_target, weights, criterion):
    token_loss = criterion(logits.transpose(1, 2), decoder_target)
    valid_mask = decoder_target.ne(0)
    valid_counts = valid_mask.sum(dim=1).clamp(min=1)
    seq_loss = (token_loss * valid_mask).sum(dim=1) / valid_counts
    return (seq_loss * weights).mean()

total_params = sum(p.numel() for p in model.parameters())
print(f"model parameters: {total_params:,}")
print(f"training on:      {DEVICE}")
print(f"train batches:    {len(train_loader)}")
print(f"val batches:      {len(val_loader)}")
print(f"model version:    {MODEL_VERSION}")

# ── helpers ──────────────────────────────────────────────

def format_duration(seconds):
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_steps = 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids  = batch["input"].to(device)
            context    = batch["context"].to(device)
            target_ids = batch["target"].to(device)
            target_cmd_types = batch["target_cmd_type"].to(device)
            weights = batch["weight"].to(device)
            decoder_input = target_ids[:, :-1]
            decoder_target = target_ids[:, 1:]
            logits, _ = model(input_ids, context, decoder_input, target_cmd_types)
            loss = weighted_sequence_loss(logits, decoder_target, weights, criterion)
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
            _, cmd_type_logits = model(input_ids, context, target_ids[:, :-1], target_cmd_types)
            preds = cmd_type_logits.argmax(dim=-1)
            correct += (preds == target_cmd_types).sum().item()
            total += target_cmd_types.size(0)
    return correct / total if total > 0 else 0.0

def notify(
    title,
    message,
    level="info",
    event="completed",
    best_val_loss=0,
    step=0,
    epoch=None,
    type_acc=None,
    duration_seconds=None,
):
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
                    "model_version": MODEL_VERSION,
                    "best_val_loss": best_val_loss,
                    "total_steps": step,
                    "epochs": EPOCHS,
                    "epoch": epoch,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "decoder_layers": CFG["decoder_layers"],
                    "type_emb_dim": CFG["type_emb_dim"],
                    "dropout": CFG["dropout"],
                    "type_acc": type_acc,
                    "duration_seconds": duration_seconds,
                    "duration_hms": format_duration(duration_seconds) if duration_seconds is not None else None,
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
train_start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:

        input_ids  = batch["input"].to(DEVICE)
        context    = batch["context"].to(DEVICE)
        target_ids = batch["target"].to(DEVICE)
        target_cmd_types = batch["target_cmd_type"].to(DEVICE)
        weights = batch["weight"].to(DEVICE)

        optimizer.zero_grad()

        decoder_input  = target_ids[:, :-1]   # drop last token
        decoder_target = target_ids[:, 1:]    # drop first token (<s>)
        alpha = get_alpha(epoch)

        logits, cmd_type_logits = model(input_ids, context, decoder_input, target_cmd_types)
        cmd_loss = weighted_sequence_loss(logits, decoder_target, weights, criterion)
        type_loss = cmd_type_criterion(cmd_type_logits, target_cmd_types)
        loss = cmd_loss + alpha * type_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - train_start_time
            print(
                f"epoch {epoch+1} | step {step} | loss {loss.item():.4f} "
                f"| cmd {cmd_loss.item():.4f} | type {type_loss.item():.4f} "
                f"| alpha {alpha:.2f} | lr {lr:.6f} | elapsed {format_duration(elapsed)}"
            )

        if step % EVAL_EVERY == 0 and step > 0:
            val_loss = evaluate(model, val_loader, criterion, DEVICE)
            type_acc = evaluate_cmd_type_accuracy(model, val_loader, DEVICE)
            elapsed = time.time() - train_start_time
            print(
                f"epoch {epoch+1} | step {step} | train {loss.item():.4f} "
                f"| val {val_loss:.4f} | type_acc {type_acc:.4f} | elapsed {format_duration(elapsed)}"
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
    elapsed = time.time() - train_start_time
    print(f"epoch {epoch+1} complete | elapsed {format_duration(elapsed)}")
    notify(
        title=f"mnesh epoch {epoch+1} complete",
        message=(
            f"epoch {epoch+1}/{EPOCHS} finished at step {step} "
            f"with train loss {loss.item():.4f} in {format_duration(elapsed)}"
        ),
        level="info",
        event="epoch_completed",
        best_val_loss=best_val_loss,
        step=step,
        epoch=epoch + 1,
        duration_seconds=elapsed,
    )

print("training complete")
total_duration = time.time() - train_start_time

notify(
    title="mnesh training complete",
    message=(
        f"4 epochs done. best val loss: {best_val_loss:.4f} "
        f"at step {step} in {format_duration(total_duration)}"
    ),
    level="info",
    event="completed",
    best_val_loss=best_val_loss,
    step=step,
    epoch=EPOCHS,
    duration_seconds=total_duration,
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
    "model_version":   MODEL_VERSION,
    "cfg":             CFG,
    "final_train_loss": loss.item(),
    "train_duration_seconds": total_duration,
    "train_duration_hms": format_duration(total_duration),
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
print(f"train duration:   {report['train_duration_hms']}")
print(f"report saved →    {report_path}")
print("────────────────────────────────────────────\n")

# auto stop pod when done
# pod_id = os.environ.get("RUNPOD_POD_ID", "")
# if pod_id:
#     print(f"stopping pod {pod_id}...")
#     subprocess.run(["runpodctl", "stop", "pod", pod_id])
