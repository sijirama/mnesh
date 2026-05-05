import os
import time

import requests
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download

from model.main import CFG, MneshModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEACON_TOKEN = os.environ.get("BEACON_TOKEN", "")

def notify(title, message, level="info", event="inference"):
    if not BEACON_TOKEN:
        print(f"[beacon] no token set, skipping notification")
        return
    try:
        r = requests.post(
            "https://beacon.sijibomi.com/emit",
            headers={
                "Authorization": f"Bearer {BEACON_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "title": title,
                "message": message,
                "source": "mnesh",
                "event": event,
                "level": level,
                "channel": "email",
            }
        )
        print(f"[beacon] {r.status_code} — {r.json()}")
    except Exception as e:
        print(f"[beacon] failed: {e}")

# load tokenizer
model_path = hf_hub_download(
    repo_id="sijirama/mnesh-unigram-tokenizer",
    filename="mnesh_unigram.model"
)
sp = spm.SentencePieceProcessor()
sp.load(model_path)

# load model
model = MneshModel(CFG).to(DEVICE)
checkpoint = torch.load("checkpoints/mnesh_best.pt", map_location=DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()
print(f"model loaded — best val loss: {checkpoint['loss']:.4f}")

# test commands
test_commands = [
    "cd ~/projects/api-service",
    "git status",
    "git add .",
    "git diff --cached",
    "python3 -m pytest tests/",
    "docker ps",
    "ls -la",
    "cat README.md",
    "git log --oneline -5",
    "git branch",
]

context = torch.tensor([[
    1,  # os: macos
    0,  # shell: zsh
    0,  # session_context: backend_dev
    1,  # cmd_type: git
    1,  # git_enabled: true
]], dtype=torch.long).to(DEVICE)

def tokenize_cmd(cmd):
    ids = sp.encode_as_ids(cmd, add_bos=False, add_eos=False)
    ids = ids[:32]
    ids = ids + [0] * (32 - len(ids))
    return ids

input_ids = torch.tensor(
    [[tokenize_cmd(cmd) for cmd in test_commands]],
    dtype=torch.long
).to(DEVICE)

PAD_ID = 0
BOS_ID = sp.piece_to_id("<s>")
EOS_ID = sp.piece_to_id("</s>")
REPETITION_PENALTY = 2.0

with torch.no_grad():
    tok_emb, ctx_vec = model.embedding(input_ids, context)
    cmd_vecs    = model.inner_gru(tok_emb)
    session_vec = model.outer_gru(cmd_vecs)
    seed        = model.projector(session_vec, ctx_vec)

    generated = [BOS_ID]
    hidden    = None
    max_tokens = 32

    for step in range(max_tokens):
        current_token = torch.tensor([[generated[-1]]], dtype=torch.long).to(DEVICE)
        embedded      = model.decoder.embedding(current_token)
        seed_step     = seed.unsqueeze(1)
        rnn_input     = torch.cat([embedded, seed_step], dim=-1)
        output, hidden = model.decoder.rnn(rnn_input, hidden)
        logits        = model.decoder.output_projection(output.squeeze(1))

        # mask special tokens
        logits[0, PAD_ID] = -float("inf")
        logits[0, BOS_ID] = -float("inf")

        # repetition penalty
        for token_id in set(generated):
            logits[0, token_id] /= REPETITION_PENALTY

        next_token = torch.argmax(logits, dim=-1).item()
        print(f"step {step}: token_id={next_token} piece={sp.id_to_piece(next_token)}")

        if next_token == EOS_ID:
            print("hit </s> — stopping")
            break

        generated.append(next_token)

suggestion = sp.decode_ids(generated[1:])
print(f"\nrecent session:")
for cmd in test_commands:
    print(f"  {cmd}")
print(f"\nmnesh suggests: {suggestion}")

# send notification
notify(
    title="mnesh inference complete",
    message=f"session: {test_commands[-1]} → suggestion: {suggestion}",
    level="info",
    event="completed"
)
