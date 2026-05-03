
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.dataset import MneshDatasetV1
from model.main import CFG, MneshModel

# hyperparameters
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
EPOCHS        = 3
EVAL_EVERY    = 500   # evaluate every N steps
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# setup
dataset   = MneshDatasetV1()
loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model     = MneshModel(CFG)
model     = model.to(DEVICE)
criterion = CrossEntropyLoss(ignore_index=0)
optimizer = Adam(model.parameters(),lr=LEARNING_RATE)

total_params = sum(p.numel() for p in model.parameters())
print(f"model parameters: {total_params:,}")
print(f"training on: {DEVICE}")
print(f"total batches per epoch: {len(loader)}")
