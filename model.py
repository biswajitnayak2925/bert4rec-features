import torch
import torch.nn as nn

import config as cfg


class ContentFusionEmbedding(nn.Module):
    def __init__(self, num_items, mask_token, pad_token, variant,
                 genre_mat=None, meta=None, plot_mat=None):
        super().__init__()
        self.variant = variant
        vocab_size = num_items + 2  # PAD, items, MASK

        self.id_emb = nn.Embedding(vocab_size, cfg.HIDDEN_SIZE, padding_idx=pad_token)

        use_genre = variant in ("M2_genre", "M5_all") and genre_mat is not None
        use_meta = variant in ("M3_metadata", "M5_all") and meta is not None
        use_plot = variant in ("M4_plot", "M5_all") and plot_mat is not None

        self.use_genre, self.use_meta, self.use_plot = use_genre, use_meta, use_plot

        extra_dim = 0

        if use_genre:
            self.register_buffer("genre_mat", torch.tensor(genre_mat), persistent=False)
            self.genre_proj = nn.Linear(genre_mat.shape[1], cfg.GENRE_DIM)
            extra_dim += cfg.GENRE_DIM

        if use_meta:
            self.register_buffer("director_ids", torch.tensor(meta["director_ids"]), persistent=False)
            self.register_buffer("lead_ids", torch.tensor(meta["lead_ids"]), persistent=False)
            self.register_buffer("runtime_norm", torch.tensor(meta["runtime_norm"]), persistent=False)
            dir_dim = cfg.METADATA_DIM // 2
            lead_dim = cfg.METADATA_DIM // 2
            self.director_emb = nn.Embedding(meta["num_directors"] + 1, dir_dim, padding_idx=0)
            self.lead_emb = nn.Embedding(meta["num_leads"] + 1, lead_dim, padding_idx=0)
            self.meta_proj = nn.Linear(dir_dim + lead_dim + 1, cfg.METADATA_DIM)
            extra_dim += cfg.METADATA_DIM

        if use_plot:
            self.register_buffer("plot_mat", torch.tensor(plot_mat), persistent=False)
            self.plot_proj = nn.Linear(plot_mat.shape[1], cfg.PLOT_PROJ_DIM)
            extra_dim += cfg.PLOT_PROJ_DIM

        self.extra_dim = extra_dim
        if extra_dim > 0:
            self.fuse = nn.Linear(cfg.HIDDEN_SIZE + extra_dim, cfg.HIDDEN_SIZE)
        else:
            self.fuse = None

        self.dropout = nn.Dropout(cfg.DROPOUT)

    def forward(self, item_ids):
        x = self.id_emb(item_ids)
        parts = [x]

        if self.use_genre:
            g = self.genre_mat[item_ids]                
            parts.append(self.genre_proj(g))

        if self.use_meta:
            d = self.director_emb(self.director_ids[item_ids])
            l = self.lead_emb(self.lead_ids[item_ids])
            r = self.runtime_norm[item_ids].unsqueeze(-1)
            m = torch.cat([d, l, r], dim=-1)
            parts.append(self.meta_proj(m))

        if self.use_plot:
            p = self.plot_mat[item_ids]
            parts.append(self.plot_proj(p))

        if len(parts) > 1:
            fused = torch.cat(parts, dim=-1)
            out = self.fuse(fused)
        else:
            out = x

        return self.dropout(out)


class BERT4Rec(nn.Module):
    def __init__(self, num_items, mask_token, pad_token, variant,
                 genre_mat=None, meta=None, plot_mat=None,
                 max_seq_len=None):
        super().__init__()
        if max_seq_len is None:
            max_seq_len = cfg.MAX_SEQ_LEN
        self.num_items = num_items
        self.mask_token = mask_token
        self.pad_token = pad_token
        vocab_size = num_items + 2

        self.content_emb = ContentFusionEmbedding(
            num_items, mask_token, pad_token, variant, genre_mat, meta, plot_mat
        )
        self.pos_emb = nn.Embedding(max_seq_len, cfg.HIDDEN_SIZE)
        self.layernorm = nn.LayerNorm(cfg.HIDDEN_SIZE)
        self.dropout = nn.Dropout(cfg.DROPOUT)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.HIDDEN_SIZE,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=cfg.INNER_SIZE,
            dropout=cfg.DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.NUM_LAYERS)
        self.output_bias = nn.Parameter(torch.zeros(num_items + 1)) 

    def encode(self, item_ids):
        batch, seq_len = item_ids.shape
        device = item_ids.device

        emb = self.content_emb(item_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
        emb = emb + self.pos_emb(positions)
        emb = self.dropout(self.layernorm(emb))

        pad_mask = (item_ids == self.pad_token)  
        hidden = self.encoder(emb, src_key_padding_mask=pad_mask) 
        return hidden

    def project(self, hidden_subset):
        item_weight = self.content_emb.id_emb.weight[: self.num_items + 1] 
        return hidden_subset @ item_weight.t() + self.output_bias

    def forward(self, item_ids):
        hidden = self.encode(item_ids)
        return self.project(hidden)


def masked_ce_loss(logits, labels, pad_token=0):
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))


if __name__ == "__main__":
    import numpy as np
    num_items, mask_token, pad_token = 199, 200, 0
    genre_mat = np.random.rand(num_items + 2, 8).astype(np.float32)
    meta = {
        "director_ids": np.random.randint(0, 30, num_items + 2),
        "lead_ids": np.random.randint(0, 40, num_items + 2),
        "runtime_norm": np.random.randn(num_items + 2).astype(np.float32),
        "num_directors": 30, "num_leads": 40,
    }
    plot_mat = np.random.rand(num_items + 2, 128).astype(np.float32)

    for variant in cfg.VARIANTS:
        model = BERT4Rec(num_items, mask_token, pad_token, variant, genre_mat, meta, plot_mat)
        batch = torch.randint(0, num_items + 2, (4, cfg.MAX_SEQ_LEN))
        out = model(batch)
        print(variant, "logits shape:", out.shape)
