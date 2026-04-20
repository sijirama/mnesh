from huggingface_hub import hf_hub_download
import sentencepiece as spm

print("loading tokenizer from HF...")
model_path = hf_hub_download(
    repo_id="sijirama/mnesh-unigram-tokenizer",
    filename="mnesh_unigram.model"
)

sp = spm.SentencePieceProcessor()
sp.load(model_path)

print(f"vocab size: {sp.get_piece_size()}")

test_cmds = [
    "git commit -m 'fix: auth refactor'",
    "docker build -t web-app:dev .",
    "kubectl get pods -n production",
    "cd ~/projects/api-service && ls -la",
    "grep -rn 'secret' config/",
    "pip install -r requirements.txt",
    "ssh root@161.63.126.193 'systemctl list-units --failed'",
    "git log --oneline -10",
    "cat /etc/nginx/nginx.conf | grep -i server",
    "python3 -m pytest tests/ -v --tb=short",
]

print("\n--- tokenisation spot check ---\n")
for cmd in test_cmds:
    tokens = sp.encode_as_pieces(cmd)
    ids = sp.encode_as_ids(cmd)
    decoded = sp.decode_pieces(tokens)
    print(f"cmd:     {cmd}")
    print(f"tokens:  {tokens}")
    print(f"ids:     {ids}")
    print(f"decoded: {decoded}")
    print(f"pieces:  {len(tokens)}")
    print()
