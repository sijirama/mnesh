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

lengths = []
for example in dataset:
    tokens = sp.encode_as_ids(example["cmd"])
    lengths.append(len(tokens))

lengths.sort()
total = len(lengths)

print(f"min:    {min(lengths)}")
print(f"max:    {max(lengths)}")
print(f"mean:   {sum(lengths)/total:.1f}")
print(f"p50:    {lengths[int(total*0.50)]}")
print(f"p90:    {lengths[int(total*0.90)]}")
print(f"p95:    {lengths[int(total*0.95)]}")
print(f"p99:    {lengths[int(total*0.99)]}")
