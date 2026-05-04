import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download

from model.main import CFG, MneshModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# test commands — simulating a real session
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

# encode context — using sensible defaults
context = torch.tensor([[
    1,  # os: linux=0, macos=1
    0,  # shell: zsh
    0,  # session_context: backend_dev
    1,  # cmd_type: git
    1,  # git_enabled: true
]], dtype=torch.long).to(DEVICE)

# tokenise and pad each command
def tokenize_cmd(cmd):
    ids = sp.encode_as_ids(cmd, add_bos=False, add_eos=False)
    ids = ids[:32]
    ids = ids + [0] * (32 - len(ids))
    return ids

input_ids = torch.tensor(
    [[tokenize_cmd(cmd) for cmd in test_commands]],
    dtype=torch.long
).to(DEVICE)

# generate next command
with torch.no_grad():
    # encoder pass
    tok_emb, ctx_vec = model.embedding(input_ids, context)
    cmd_vecs    = model.inner_gru(tok_emb)
    session_vec = model.outer_gru(cmd_vecs)
    seed        = model.projector(session_vec, ctx_vec)

    # autoregressive decoder
    bos_id = sp.piece_to_id("<s>")
    eos_id = sp.piece_to_id("</s>")

    generated = [bos_id]
    hidden = None
    max_tokens = 32
    temperature = 0.3

    for _ in range(max_tokens):
        current_token = torch.tensor([[generated[-1]]], dtype=torch.long).to(DEVICE)
        embedded = model.decoder.embedding(current_token)
        rnn_input = torch.cat([embedded, seed.unsqueeze(1)], dim=-1)
        output, hidden = model.decoder.rnn(rnn_input, hidden)
        logits = model.decoder.output_projection(output.squeeze(1))
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        if next_token == eos_id:
            break
        generated.append(next_token)

    suggestion = sp.decode_ids(generated[1:])  # skip <s>
    print(f"\nrecent session:")
    for cmd in test_commands:
        print(f"  {cmd}")
    print(f"\nmnesh suggests: {suggestion}")
