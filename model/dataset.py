from collections import defaultdict

import sentencepiece as spm
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset


class MneshDatasetV1(Dataset):
    def __init__(self):
        self.max_cmd_len = 32
        self.window_size = 10
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

        # INFO: load the tokenizer
        model_path = hf_hub_download(
                    repo_id="sijirama/mnesh-unigram-tokenizer",
                    filename="mnesh_unigram.model"
        )
        tk = spm.SentencePieceProcessor()
        tk.load(model_path)
        self.tk = tk

        # INFO: load the dataset
        dataset = load_dataset("sijirama/mnesh-shell-commands", split="train")

        # group into sessions 
        sessions = defaultdict(list)
        for row in dataset:
            id = row["session_id"]
            sessions[id].append(row)

        # sort the lists back in the sequence order
        for key in sessions:
            sessions[key].sort(key=lambda x:x['sequence_index'])

        self.sessions = sessions

        self.windows = []

        for session in sessions.values():
            if len(session) < self.window_size + 1:
                continue
            for i in range(len(session) - self.window_size):
                window = session[i : i + self.window_size]
                target = session[i + self.window_size]
                self.windows.append((window,target))

    def __len__(self):
        return len(self.windows)

    def _tokenize_and_pad(self, cmd: str) -> list[int]:
        ids = self.tk.encode_as_ids(cmd, add_bos=False, add_eos=False)
        ids = ids[:self.max_cmd_len]
        ids = ids + [0] * (self.max_cmd_len - len(ids))
        return ids

    def _tokenize_target(self, cmd: str) -> list[int]:
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
        window, target = self.windows[idx]

        # build input tensor (10, 32)
        input_ids = [self._tokenize_and_pad(row["cmd"]) for row in window]
        input_tensor = torch.tensor(input_ids, dtype=torch.long)

        # build context tensor (5,)
        context = self._encode_context(window[-1])
        context_tensor = torch.tensor(context, dtype=torch.long)

        # build target tensor (32,)
        target_ids = self._tokenize_target(target["cmd"])
        target_tensor = torch.tensor(target_ids, dtype=torch.long)

        return {
            "input": input_tensor,
            "context": context_tensor,
            "target": target_tensor,
        }


# INFO: test the dataset
#
# dataset = MneshDatasetV1()
# print(f"total windows: {len(dataset)}")
#
# sample = dataset[0]
# print(f"input shape:   {sample['input'].shape}")
# print(f"context shape: {sample['context'].shape}")
# print(f"target shape:  {sample['target'].shape}")
#
# print(f"input[0]:  {sample['input'][0]}")
# print(f"context:   {sample['context']}")
# print(f"target:    {sample['target']}")
