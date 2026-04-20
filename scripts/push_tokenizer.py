from huggingface_hub import HfApi, create_repo

api = HfApi()
create_repo("sijirama/mnesh-unigram-tokenizer", exist_ok=True)
api.upload_file(
    path_or_fileobj="tokenizer/mnesh_unigram.model",
    path_in_repo="mnesh_unigram.model",
    repo_id="sijirama/mnesh-unigram-tokenizer",
)
