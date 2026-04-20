import re
from collections import Counter

import sentencepiece as spm
from datasets import load_dataset
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="sijirama/mnesh-unigram-tokenizer",
    filename="mnesh_unigram.model"
)
sp = spm.SentencePieceProcessor()
sp.load(model_path)

dataset = load_dataset("sijirama/mnesh-shell-commands", split="train")

flag_counter = Counter()
operator_counter = Counter()

for example in dataset:
    cmd = example["cmd"]
    flags = re.findall(r'--[\w-]+', cmd)
    operators = re.findall(r'&&|\|\||>>|<<|\|', cmd)
    flag_counter.update(flags)
    operator_counter.update(operators)

print("top 30 flags:")
for flag, count in flag_counter.most_common(30):
    tokens = sp.encode_as_pieces(flag)
    print(f"  {flag:<25} count: {count:<8} tokens: {tokens}")

print("\noperators:")
for op, count in operator_counter.most_common():
    tokens = sp.encode_as_pieces(op)
    print(f"  {op:<10} count: {count:<8} tokens: {tokens}")
