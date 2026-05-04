import os
import pickle
import random
import re
from collections import defaultdict

import sentencepiece as spm
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

COMMIT_PATTERNS = [
    r'git commit -m ["\']',
    r'git commit --message ["\']',
    r'git commit -am ["\']',
    r'git commit -a -m ["\']',
]

class MneshDatasetV1(Dataset):
    def __init__(self, split="train"):
        self.max_cmd_len = 32
        self.window_size = 10
        self.split       = split
        self.os_map = {"linux": 0, "macos": 1}
        self.shell_map = {"zsh": 0, "bash": 1, "fish": 2}
        self.session_context_map = {
            "backend_dev": 0, "frontend_dev": 1, "devops": 2,
            "data_science": 3, "debugging": 4, "deployment": 5,
            "exploration": 6, "system_admin": 7, "open_source_contrib": 8
        }
        self.cmd_type_map = {
            "filesystem": 0, "git": 1, "process": 2, "network": 3,
            "package": 4, "docker": 5, "k8s": 6, "python": 7,
            "node": 8, "system": 9, "text_processing": 10, "ssh": 11, "misc": 12
        }

        # load tokenizer
        model_path = hf_hub_download(
            repo_id="sijirama/mnesh-unigram-tokenizer",
            filename="mnesh_unigram.model"
        )
        tk = spm.SentencePieceProcessor()
        tk.load(model_path)
        self.tk = tk

        cache_path = "data/windows_cache.pkl"

        if os.path.exists(cache_path):
            print("loading windows from cache...")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            self.train_windows = cache["train"]
            self.val_windows   = cache["val"]
            self.test_windows  = cache["test"]
            print(f"train: {len(self.train_windows):,} | val: {len(self.val_windows):,} | test: {len(self.test_windows):,}")

        else:
            print("building windows from scratch, this will take a while...")
            dataset = load_dataset("sijirama/mnesh-shell-commands", split="train")

            # group into sessions
            sessions = defaultdict(list)
            for row in dataset:
                sessions[row["session_id"]].append(row)

            # sort by sequence order
            for key in sessions:
                sessions[key].sort(key=lambda x: x['sequence_index'])

            # split session ids 80/10/10
            session_ids = list(sessions.keys())
            random.seed(42)
            random.shuffle(session_ids)
            n = len(session_ids)
            train_ids = set(session_ids[:int(n * 0.8)])
            val_ids   = set(session_ids[int(n * 0.8):int(n * 0.9)])
            test_ids  = set(session_ids[int(n * 0.9):])

            # build windows per split
            self.train_windows = []
            self.val_windows   = []
            self.test_windows  = []

            for session_id, session in sessions.items():
                if len(session) < self.window_size + 1:
                    continue
                for i in range(len(session) - self.window_size):
                    window = session[i : i + self.window_size]
                    target = session[i + self.window_size]
                    if session_id in train_ids:
                        self.train_windows.append((window, target))
                    elif session_id in val_ids:
                        self.val_windows.append((window, target))
                    else:
                        self.test_windows.append((window, target))

            # save cache
            print("saving windows to cache...")
            os.makedirs("data", exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "train": self.train_windows,
                    "val":   self.val_windows,
                    "test":  self.test_windows,
                }, f)
            print(f"train: {len(self.train_windows):,} | val: {len(self.val_windows):,} | test: {len(self.test_windows):,}")

    def _get_windows(self):
        if self.split == "train":
            return self.train_windows
        elif self.split == "val":
            return self.val_windows
        else:
            return self.test_windows

    def __len__(self):
        return len(self._get_windows())

    def is_commit_command(self, cmd):
        return any(re.match(p, cmd) for p in COMMIT_PATTERNS)

    def truncate_commit(self, cmd):
        match = re.match(r'(git commit (?:-a )?(?:-m|--message) ["\'])', cmd)
        if match:
            return match.group(1) + "</s>"
        return cmd

    def _tokenize_and_pad(self, cmd: str) -> list[int]:
        ids = self.tk.encode_as_ids(cmd, add_bos=False, add_eos=False)
        ids = ids[:self.max_cmd_len]
        ids = ids + [0] * (self.max_cmd_len - len(ids))
        return ids

    def _tokenize_target(self, cmd: str) -> list[int]:
        if self.is_commit_command(cmd):
            cmd = self.truncate_commit(cmd)
        ids = self.tk.encode_as_ids(cmd, add_bos=True, add_eos=True)
        ids = ids[:self.max_cmd_len]
        ids = ids + [0] * (self.max_cmd_len - len(ids))
        return ids

    def _encode_context(self, row) -> list[int]:
        return [
            self.os_map.get(row["os"], 0),
            self.shell_map.get(row["shell"], 0),
            self.session_context_map.get(row["session_context"], 0),
            self.cmd_type_map.get(row["cmd_type"], 0),
            1 if row["git_enabled"] else 0,
        ]

    def __getitem__(self, idx):
        window, target = self._get_windows()[idx]

        input_ids = [self._tokenize_and_pad(row["cmd"]) for row in window]
        input_tensor   = torch.tensor(input_ids, dtype=torch.long)
        context_tensor = torch.tensor(self._encode_context(window[-1]), dtype=torch.long)
        target_tensor  = torch.tensor(self._tokenize_target(target["cmd"]), dtype=torch.long)

        return {
            "input":   input_tensor,
            "context": context_tensor,
            "target":  target_tensor,
        }

# INFO: test the dataset

dataset = MneshDatasetV1(split="train")
print(f"total windows: {len(dataset)}")

sample = dataset[0]
print(f"input shape:   {sample['input'].shape}")
print(f"context shape: {sample['context'].shape}")
print(f"target shape:  {sample['target'].shape}")

print(f"input[0]:  {sample['input'][0]}")
print(f"context:   {sample['context']}")
print(f"target:    {sample['target']}")
