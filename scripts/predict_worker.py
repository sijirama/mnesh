#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model import main as v6_main  # noqa: E402
from model import v5_main  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SESSION_CONTEXT_MAP = {
    "backend_dev": 0,
    "frontend_dev": 1,
    "devops": 2,
    "data_science": 3,
    "debugging": 4,
    "deployment": 5,
    "exploration": 6,
    "system_admin": 7,
    "open_source_contrib": 8,
}
SHELL_MAP = {"zsh": 0, "bash": 1, "fish": 2}
CMD_TYPES_V5 = [
    "filesystem", "git", "process", "network", "package", "docker",
    "k8s", "python", "node", "system", "text_processing", "ssh", "misc",
]
CMD_TYPES_V6 = [
    "filesystem", "git", "process", "network", "package", "container",
    "python", "node", "system", "text_processing", "misc",
]
ECOSYSTEM_MAP = {
    "node": 0,
    "python": 1,
    "go": 2,
    "rust": 3,
    "infra": 4,
    "ruby": 5,
    "db": 6,
    "system": 7,
    "misc": 8,
}
ECOSYSTEM_PATTERNS = {
    "node": ["package.json", "node_modules", "npm ", "pnpm ", "npx ", "yarn ", "vite", "nextjs", "react"],
    "python": ["python", "pytest", "pip ", "venv", ".venv", "jupyter", "manage.py", "pyproject", "requirements.txt"],
    "go": ["go mod", "go build", "go test", "go run", "go fmt", "go vet"],
    "rust": ["cargo ", "cargo.toml", "cargo.lock", "diesel "],
    "infra": ["docker", "kubectl", "k8s", "helm ", "terraform ", "docker-compose"],
    "ruby": ["bundle exec", "gemfile", "rake ", "rails "],
    "db": ["psql", "mysql", "mongosh", "postgres", "sqlite", "redis", "sql"],
}


def load_payload():
    if len(sys.argv) < 2:
        raise SystemExit("missing JSON payload")
    return json.loads(sys.argv[1])


def choose_runtime(version):
    if version.startswith("v5"):
        return v5_main, CMD_TYPES_V5, False
    return v6_main, CMD_TYPES_V6, True


def detect_ecosystem(events):
    combined = " ".join(
        f"{event.get('cmd','')} {event.get('cwd','')} {event.get('git_branch','')}"
        for event in events
    ).lower()
    for ecosystem, patterns in ECOSYSTEM_PATTERNS.items():
        if any(pattern in combined for pattern in patterns):
            return ecosystem
    return "misc"


def detect_cmd_type(cmd):
    text = cmd.strip().lower()
    if text.startswith("git "):
        return "git"
    if text.startswith(("docker ", "docker-compose", "docker compose")):
        return "docker"
    if text.startswith(("kubectl ", "helm ")):
        return "k8s"
    if text.startswith(("python", "pytest", "pip ", "jupyter ", "conda ", "django-admin")):
        return "python"
    if text.startswith(("npm ", "pnpm ", "npx ", "yarn ", "vite ", "node ")):
        return "node"
    if text.startswith(("ssh ", "scp ")):
        return "ssh"
    if text.startswith(("systemctl ", "journalctl ", "service ", "sudo systemctl", "top", "htop", "free ", "uptime", "df ", "du ", "ncdu", "ps ", "ss ", "netstat ")):
        return "system"
    if text.startswith(("grep ", "rg ", "awk ", "sed ", "jq ", "cut ", "sort ", "uniq ", "head ", "tail ")):
        return "text_processing"
    if text.startswith(("curl ", "wget ", "ping ", "dig ", "nslookup ")):
        return "network"
    if text.startswith(("apt ", "brew ", "yum ", "dnf ", "apk ", "pacman ")):
        return "package"
    if text.startswith(("cd ", "ls", "pwd", "mkdir ", "rm ", "cp ", "mv ", "find ")):
        return "filesystem"
    if text.startswith(("go ", "cargo ")):
        return "misc"
    return "process" if "|" in text else "misc"


def normalize_cmd_type(cmd_type):
    if cmd_type in {"docker", "k8s"}:
        return "container"
    if cmd_type in {"system", "ssh"}:
        return "system"
    return cmd_type


def detect_session_context(ecosystem, events):
    combined = " ".join(event.get("cmd", "") for event in events).lower()
    if ecosystem == "node":
        return "frontend_dev"
    if ecosystem == "python":
        return "data_science" if any(token in combined for token in ["jupyter", "pandas", "scikit", "train.py", "evaluate.py"]) else "backend_dev"
    if ecosystem in {"go", "rust"}:
        return "backend_dev"
    if ecosystem == "infra":
        return "deployment" if any(token in combined for token in ["kubectl", "helm", "terraform"]) else "devops"
    if ecosystem in {"db", "system"}:
        return "system_admin"
    return "exploration"


def build_context(events, is_v6):
    last = events[-1] if events else {}
    ecosystem = detect_ecosystem(events)
    session_context = detect_session_context(ecosystem, events)
    raw_cmd_type = detect_cmd_type(last.get("cmd", ""))
    git_enabled = any(event.get("git_branch") for event in events) or any("git " in event.get("cmd", "") for event in events)

    base = [
        0,  # linux default
        SHELL_MAP.get(last.get("shell", "zsh"), 0),
        SESSION_CONTEXT_MAP.get(session_context, 6),
    ]

    if is_v6:
        coarse = normalize_cmd_type(raw_cmd_type)
        cmd_map = {name: idx for idx, name in enumerate(CMD_TYPES_V6)}
        base.extend([
            cmd_map.get(coarse, cmd_map["misc"]),
            1 if git_enabled else 0,
            ECOSYSTEM_MAP.get(ecosystem, ECOSYSTEM_MAP["misc"]),
        ])
    else:
        cmd_map = {name: idx for idx, name in enumerate(CMD_TYPES_V5)}
        base.extend([
            cmd_map.get(raw_cmd_type, cmd_map["misc"]),
            1 if git_enabled else 0,
        ])
    return base, ecosystem, session_context


def tokenize_cmd(sp, cmd, max_len):
    ids = sp.encode_as_ids(cmd, add_bos=False, add_eos=False)
    return ids[:max_len] + [0] * (max_len - len(ids[:max_len]))


def prepare_window(events, cfg):
    commands = [event.get("cmd", "") for event in events][-cfg["window_size"]:]
    if len(commands) < cfg["window_size"]:
        commands = [""] * (cfg["window_size"] - len(commands)) + commands
    return commands


def predict(payload):
    model_dir = Path(payload["model_dir"])
    version = (model_dir / "VERSION").read_text().strip()
    runtime, cmd_names, is_v6 = choose_runtime(version)

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_dir / "mnesh_unigram.model"))

    model = runtime.MneshModel(runtime.CFG).to(DEVICE)
    checkpoint = torch.load(model_dir / "mnesh_best.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    events = payload["events"]
    commands = prepare_window(events, runtime.CFG)
    context_values, ecosystem, session_context = build_context(events, is_v6)
    input_ids = torch.tensor(
        [[tokenize_cmd(sp, cmd, runtime.CFG["max_cmd_len"]) for cmd in commands]],
        dtype=torch.long,
        device=DEVICE,
    )
    context = torch.tensor([context_values], dtype=torch.long, device=DEVICE)

    bos_id = sp.piece_to_id("<s>")
    eos_id = sp.piece_to_id("</s>")
    pad_id = 0

    with torch.no_grad():
        tok_emb, ctx_vec = model.embedding(input_ids, context)
        cmd_vecs = model.inner_gru(tok_emb, input_ids)
        _, session_vec = model.outer_gru(cmd_vecs)
        type_logits = model.cmd_type_head(session_vec)
        type_probs = torch.softmax(type_logits, dim=-1)
        top_probs, top_ids = torch.topk(type_probs, 3, dim=-1)
        pred_type_ids = top_ids[:, :1].squeeze(1)
        type_vec = model.decoder.type_embedding(pred_type_ids)
        seed = model.projector(session_vec, ctx_vec, type_vec)
        hidden = model.decoder.seed_projection(seed)
        hidden = hidden.view(1, model.decoder.num_layers, model.decoder.hidden_size).transpose(0, 1).contiguous()

        generated = [bos_id]
        for _ in range(payload.get("max_tokens", 32)):
            current = torch.tensor([[generated[-1]]], dtype=torch.long, device=DEVICE)
            token_emb = model.decoder.embedding(current)
            raw_type = model.decoder.type_embedding(pred_type_ids).unsqueeze(1)
            gate_input = torch.cat([token_emb, raw_type], dim=-1)
            gate = model.decoder.gate(gate_input)
            type_emb = model.decoder.type_adapter(raw_type)
            fused = token_emb + gate * type_emb
            output, hidden = model.decoder.rnn(fused, hidden)
            logits = model.decoder.output_projection(output.squeeze(1))
            logits[0, pad_id] = -float("inf")
            logits[0, bos_id] = -float("inf")
            for token_id in set(generated):
                logits[0, token_id] /= payload.get("repetition_penalty", 2.0)
            next_token = torch.argmax(logits, dim=-1).item()
            if next_token == eos_id:
                break
            generated.append(next_token)

    return {
        "model_version": version,
        "checkpoint_loss": checkpoint.get("loss"),
        "predicted_cmd_type": cmd_names[top_ids[0, 0].item()],
        "top_cmd_types": [
            {"name": cmd_names[idx.item()], "prob": round(prob.item(), 4)}
            for idx, prob in zip(top_ids[0], top_probs[0])
        ],
        "session_context": session_context,
        "ecosystem": ecosystem,
        "context": context_values,
        "window_commands": commands,
        "suggestion": sp.decode_ids(generated[1:]),
    }


if __name__ == "__main__":
    print(json.dumps(predict(load_payload())))
