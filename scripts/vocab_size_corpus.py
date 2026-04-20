import re
from collections import Counter

from datasets import load_dataset

dataset = load_dataset("sijirama/mnesh-shell-commands", split="train")

words = Counter()
for example in dataset:
    tokens = re.findall(r'\S+', example['cmd'])
    words.update(tokens)

print(f"Unique raw tokens: {len(words)}")
print(f"Top 20: {words.most_common(20)}")
print(f"Tokens appearing once: {sum(1 for v in words.values() if v == 1)}")
