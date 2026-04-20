import os

import sentencepiece as spm
from datasets import load_dataset

os.makedirs("tokenizer", exist_ok=True)

print("loading dataset...")
dataset = load_dataset("sijirama/mnesh-shell-commands", split="train")
print(f"loaded {len(dataset)} examples")

def cmd_iterator(dataset):
    for example in dataset:
        yield example["cmd"]

user_defined = [
    "<mask>",
    # operators
    "&&", "||", ">>", "<<", "2>&1", "~/",
    "▁&&", "▁||", "▁>>", "▁|", "▁~/",
    # flags — both with and without leading space
    "--oneline",        "▁--oneline",
    "--set-upstream",   "▁--set-upstream",
    "--resume",         "▁--resume",
    "--dangerously-skip-permissions", "▁--dangerously-skip-permissions",
    "--save-dev",       "▁--save-dev",
    "--failed",         "▁--failed",
    "--sort-by",        "▁--sort-by",
    "--soft",           "▁--soft",
    "--hard",           "▁--hard",
    "--title",          "▁--title",
    "--noEmit",         "▁--noEmit",
    "--nocapture",      "▁--nocapture",
]

print("training tokenizer...")
spm.SentencePieceTrainer.train(
    sentence_iterator=cmd_iterator(dataset),
    model_prefix="tokenizer/mnesh_unigram",
    vocab_size=18000,
    model_type="unigram",
    character_coverage=0.9995,
    byte_fallback=True,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>",
    user_defined_symbols=user_defined,
    input_sentence_size=5000000,
    shuffle_input_sentence=True,
)

print("done — tokenizer/mnesh_unigram.model")
