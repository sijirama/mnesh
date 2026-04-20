from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="data/shell_commands.jsonl",
    split="train"
)

dataset.push_to_hub("sijirama/mnesh-shell-commands")
