import torch
import sentencepiece as spm
from huggingface_hub import hf_hub_download

from model.main import CFG, MneshModel

DEVICE = "cpu"
CMD_TYPE_NAMES = [
    "filesystem", "git", "process", "network", "package", "docker",
    "k8s", "python", "node", "system", "text_processing", "ssh", "misc",
]


def load_model():
    model_path = hf_hub_download(
        repo_id="sijirama/mnesh-unigram-tokenizer",
        filename="mnesh_unigram.model",
    )
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    model = MneshModel(CFG).to(DEVICE)
    checkpoint = torch.load("checkpoints/mnesh_best.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, sp


def tok(sp, cmd):
    ids = sp.encode_as_ids(cmd, add_bos=False, add_eos=False)
    return ids[:32] + [0] * (32 - len(ids))


def run_session(model, sp, input_ids, context, temp=0.8, top_k=5, rep_pen=2.0):
    bos_id = sp.piece_to_id("<s>")
    eos_id = sp.piece_to_id("</s>")
    pad_id = 0

    with torch.no_grad():
        tok_emb, ctx_vec = model.embedding(input_ids, context)
        cmd_vecs = model.inner_gru(tok_emb, input_ids)
        outer_outputs = model.outer_gru(cmd_vecs)
        session_vec, attention_weights = model.attention_pool(outer_outputs)

        type_logits = model.cmd_type_head(session_vec)
        type_probs = torch.softmax(type_logits, dim=-1)
        top_type_probs, top_type_ids = torch.topk(type_probs, 3, dim=-1)
        predicted_type_ids = top_type_ids[:, :1].squeeze(1)
        type_vec = model.decoder.type_embedding(predicted_type_ids)
        seed = model.projector(session_vec, ctx_vec, type_vec)

        hidden = model.decoder.seed_projection(seed)
        hidden = hidden.view(input_ids.size(0), model.decoder.num_layers, model.decoder.hidden_size).transpose(0, 1).contiguous()
        generated = [bos_id]

        for _ in range(32):
            current = torch.tensor([[generated[-1]]], dtype=torch.long, device=DEVICE)
            token_embedded = model.decoder.embedding(current)
            raw_type_embedded = model.decoder.type_embedding(predicted_type_ids).unsqueeze(1)
            gate_input = torch.cat([token_embedded, raw_type_embedded], dim=-1)
            gate = model.decoder.gate(gate_input)
            type_embedded = model.decoder.type_adapter(raw_type_embedded)
            fused = token_embedded + gate * type_embedded
            output, hidden = model.decoder.rnn(fused, hidden)
            logits = model.decoder.output_projection(output.squeeze(1))

            logits[0, pad_id] = -float("inf")
            logits[0, bos_id] = -float("inf")

            for token_id in set(generated):
                logits[0, token_id] /= rep_pen

            if top_k == 1:
                next_token = torch.argmax(logits, dim=-1).item()
            else:
                logits = logits / temp
                values, indices = torch.topk(logits, top_k, dim=-1)
                probs = torch.softmax(values, dim=-1)
                pick = torch.multinomial(probs[0], 1).item()
                next_token = indices[0, pick].item()

            if next_token == eos_id:
                break

            generated.append(next_token)

        text = sp.decode_ids(generated[1:])
        top_types = [
            (CMD_TYPE_NAMES[idx.item()], round(prob.item(), 4))
            for idx, prob in zip(top_type_ids[0], top_type_probs[0])
        ]
        attn = [round(weight, 4) for weight in attention_weights[0, :, 0].tolist()]
        return top_types, attn, text


def main():
    model, sp = load_model()

    sessions = [
        ("git", [
            "git status", "git add .", "git diff", "git log --oneline", "git branch",
            "git stash", "git pull origin main", "git checkout feat/auth",
            "git rebase main", "git push",
        ], [1, 0, 0, 1, 1]),
        ("docker", [
            "docker ps", "docker images", "docker build -t app .", "docker run -d app",
            "docker logs app", "docker exec -it app bash", "docker stop app",
            "docker rm app", "docker pull nginx", "docker compose up",
        ], [0, 1, 2, 5, 0]),
        ("python", [
            "python3 -m pytest tests/", "python3 manage.py migrate",
            "python3 -m pip install -r requirements.txt", "python3 app.py",
            "python3 -m unittest discover", "python3 script.py",
            "python3 -m venv venv", "python3 setup.py install",
            "python3 -m pip freeze", "python3 -m http.server 8000",
        ], [0, 0, 3, 7, 0]),
        ("sysadmin", [
            "ssh root@server1", "uptime", "df -h", "top -bn1 | head -20",
            "systemctl status nginx", "journalctl -u nginx -n 50",
            "netstat -tlnp", "cat /var/log/syslog | tail -20",
            "ps aux | grep python", "free -m",
        ], [0, 1, 7, 11, 0]),
        ("frontend", [
            "cd ~/projects/web-app", "npm install", "npm run dev", "npm run build",
            "npm test", "npm run lint", "git status", "git add .",
            "git commit -m \"fix: button styling\"", "git push origin main",
        ], [1, 0, 1, 1, 1]),
    ]

    for name, cmds, ctx in sessions:
        ctx_t = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
        inp = torch.tensor([[tok(sp, cmd) for cmd in cmds]], dtype=torch.long, device=DEVICE)

        greedy_types, greedy_attn, greedy_text = run_session(model, sp, inp, ctx_t, temp=0.1, top_k=1)
        sample_types, sample_attn, sample_text1 = run_session(model, sp, inp, ctx_t, temp=0.8, top_k=5)
        _, _, sample_text2 = run_session(model, sp, inp, ctx_t, temp=0.8, top_k=5)

        print(f"\n=== {name.upper()} ===")
        print("top cmd_types:", greedy_types)
        print("attention:", greedy_attn)
        print("greedy:", greedy_text)
        print("sample:", sample_text1)
        print("sample:", sample_text2)


if __name__ == "__main__":
    main()
