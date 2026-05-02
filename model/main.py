import torch
import torch.nn as nn

from model.dataset import MneshDatasetV1  # MneshDatasetV1

OS_CLASSES          = 2
SHELL_CLASSES       = 3
SESSION_CTX_CLASSES = 9
CMD_TYPE_CLASSES    = 13
GIT_CLASSES         = 2

class MneshEmbedding(nn.Module):
    def __init__(self, vocab_size, token_emb_dim, context_emb_dim):
        super().__init__()
        self.tok_emb     = nn.Embedding(vocab_size, token_emb_dim)
        self.os_emb      = nn.Embedding(OS_CLASSES, context_emb_dim)
        self.shell_emb   = nn.Embedding(SHELL_CLASSES, context_emb_dim)
        self.ctx_emb     = nn.Embedding(SESSION_CTX_CLASSES, context_emb_dim)
        self.cmd_emb     = nn.Embedding(CMD_TYPE_CLASSES, context_emb_dim)
        self.git_emb     = nn.Embedding(GIT_CLASSES, context_emb_dim)

    def forward(self, token_ids, context):

        tok = self.tok_emb(token_ids)          # (batch, window_size(10), max_cmd_len(32), token_emb_dim)

        # context embeddings - each column of context is one feature
        os_e   = self.os_emb(context[:, 0])    # (batch, context_emb_dim)
        sh_e   = self.shell_emb(context[:, 1]) # (batch, context_emb_dim)
        ctx_e  = self.ctx_emb(context[:, 2])   # (batch, context_emb_dim)
        cmd_e  = self.cmd_emb(context[:, 3])   # (batch, context_emb_dim)
        git_e  = self.git_emb(context[:, 4])   # (batch, context_emb_dim)

        # concatenate all context embeddings into one vector
        ctx_vec = torch.cat([os_e, sh_e, ctx_e, cmd_e, git_e], dim=-1)  # (batch, 5 * context_emb_dim (80ish))

        return tok, ctx_vec

class MneshInnerGRU(nn.Module):
    def __init__(self, input_size,  hidden_size, batch_first=True):
        super().__init__()


class MneshModel(nn.Module):
    def __init__(self, vocab_size, token_emb_dim, inner_hidden, outer_hidden, context_emb_dim):
        super().__init__()

    def forward(self, input, context):
        pass
