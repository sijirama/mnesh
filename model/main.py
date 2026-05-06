import torch
import torch.nn as nn

# config — all hyperparameters in one place
CFG = {
    "vocab_size":       18000,
    "token_emb_dim":    128,
    "type_emb_dim":     64,
    "context_emb_dim":  16,
    "inner_hidden":     256,
    "outer_hidden":     512,
    "window_size":      10,
    "max_cmd_len":      32,
    "context_dim":      16 * 5,  # 80
    "dropout":          0.2,
}

OS_CLASSES          = 2
SHELL_CLASSES       = 3
SESSION_CTX_CLASSES = 9
CMD_TYPE_CLASSES    = 13
GIT_CLASSES         = 2

class MneshEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb   = nn.Embedding(cfg["vocab_size"], cfg["token_emb_dim"], padding_idx=0)
        self.os_emb    = nn.Embedding(OS_CLASSES, cfg["context_emb_dim"])
        self.shell_emb = nn.Embedding(SHELL_CLASSES, cfg["context_emb_dim"])
        self.ctx_emb   = nn.Embedding(SESSION_CTX_CLASSES, cfg["context_emb_dim"])
        self.cmd_emb   = nn.Embedding(CMD_TYPE_CLASSES, cfg["context_emb_dim"])
        self.git_emb   = nn.Embedding(GIT_CLASSES, cfg["context_emb_dim"])
        self.dropout   = nn.Dropout(cfg["dropout"])

    def forward(self, token_ids, context):
        tok   = self.dropout(self.tok_emb(token_ids))
        os_e  = self.os_emb(context[:, 0])
        sh_e  = self.shell_emb(context[:, 1])
        ctx_e = self.ctx_emb(context[:, 2])
        cmd_e = self.cmd_emb(context[:, 3])
        git_e = self.git_emb(context[:, 4])
        ctx_vec = torch.cat([os_e, sh_e, ctx_e, cmd_e, git_e], dim=-1)
        return tok, ctx_vec


class MneshInnerGRU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg["inner_hidden"]
        self.window_size = cfg["window_size"]
        self.max_cmd_len = cfg["max_cmd_len"]
        self.rnn = nn.GRU(cfg["token_emb_dim"], cfg["inner_hidden"], batch_first=True)
        self.layer_norm = nn.LayerNorm(cfg["inner_hidden"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x, token_ids):
        batch_size = x.size(0)
        x = x.view(batch_size * self.window_size, self.max_cmd_len, -1)
        flat_token_ids = token_ids.view(batch_size * self.window_size, self.max_cmd_len)
        lengths = flat_token_ids.ne(0).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        hidden = hidden.squeeze(0)
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)
        hidden = hidden.view(batch_size, self.window_size, -1)
        return hidden


class MneshOutterGRU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg["outer_hidden"]
        self.rnn = nn.GRU(cfg["inner_hidden"], cfg["outer_hidden"], batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(cfg["outer_hidden"] * 2)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        outputs, _ = self.rnn(x)
        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)
        return outputs


class MneshAttentionPool(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.score = nn.Linear(cfg["outer_hidden"] * 2, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled, weights


class MneshContextProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(
            cfg["outer_hidden"] * 2 + cfg["context_dim"] + cfg["type_emb_dim"],
            cfg["outer_hidden"]
        )
        self.layer_norm = nn.LayerNorm(cfg["outer_hidden"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, session_vector, context_vector, type_vector):
        full_vec = torch.cat([session_vector, context_vector, type_vector], dim=-1)
        out = self.linear(full_vec)
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out


class MneshDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["token_emb_dim"], padding_idx=0)
        self.type_embedding = nn.Embedding(CMD_TYPE_CLASSES, cfg["type_emb_dim"])
        self.rnn = nn.GRU(
            cfg["token_emb_dim"] + cfg["type_emb_dim"],
            cfg["outer_hidden"],
            batch_first=True
        )
        self.seed_projection = nn.Linear(cfg["outer_hidden"], cfg["outer_hidden"])
        self.output_projection = nn.Linear(cfg["outer_hidden"], cfg["vocab_size"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, target_ids, seed, cmd_type_ids):
        token_embedded = self.dropout(self.embedding(target_ids))
        type_embedded = self.type_embedding(cmd_type_ids).unsqueeze(1)
        type_embedded = type_embedded.expand(-1, target_ids.size(1), -1)
        embedded = torch.cat([token_embedded, type_embedded], dim=-1)
        h0 = self.seed_projection(seed).unsqueeze(0)
        output, _ = self.rnn(embedded, h0)
        logits = self.output_projection(output)
        return logits


class MneshCmdTypeHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(cfg["outer_hidden"] * 2, CMD_TYPE_CLASSES)

    def forward(self, session_vec):
        return self.linear(session_vec)


class MneshModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding  = MneshEmbedding(cfg)
        self.inner_gru  = MneshInnerGRU(cfg)
        self.outer_gru  = MneshOutterGRU(cfg)
        self.attention_pool = MneshAttentionPool(cfg)
        self.projector  = MneshContextProjector(cfg)
        self.decoder    = MneshDecoder(cfg)
        self.cmd_type_head = MneshCmdTypeHead(cfg)

    def forward(self, input_ids, context, target_ids, cmd_type_ids):
        tok_emb, ctx_vec = self.embedding(input_ids, context)
        cmd_vecs = self.inner_gru(tok_emb, input_ids)
        outer_outputs = self.outer_gru(cmd_vecs)
        session_vec, _ = self.attention_pool(outer_outputs)
        type_vec = self.decoder.type_embedding(cmd_type_ids)
        seed = self.projector(session_vec, ctx_vec, type_vec)
        logits = self.decoder(target_ids, seed, cmd_type_ids)
        cmd_type_logits = self.cmd_type_head(session_vec)
        return logits, cmd_type_logits
