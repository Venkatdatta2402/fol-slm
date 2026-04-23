import torch
import torch.nn as nn
import math


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.last_cross_attn_weights = None

    def forward(self, x, encoder_out, tgt_mask=None, memory_key_padding_mask=None):
        # Masked self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention over frozen encoder output
        attn_out, cross_attn_weights = self.cross_attn(
            x, encoder_out, encoder_out,
            key_padding_mask=memory_key_padding_mask,
            average_attn_weights=False,  # (B, n_heads, T_dec, T_enc)
        )
        self.last_cross_attn_weights = cross_attn_weights
        x = self.norm2(x + self.dropout(attn_out))

        # Feed-forward
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        encoder_d_model: int = 768,  # T5-base hidden size
    ):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Project encoder dim -> decoder dim if they differ
        self.encoder_proj = (
            nn.Linear(encoder_d_model, d_model)
            if encoder_d_model != d_model else nn.Identity()
        )

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.cross_attn_weights = None

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt_ids, encoder_out, encoder_attention_mask=None,
                proof_self_attn_mask=None, proof_cross_attn_mask=None, proof_layer_start=None,
                raw_memory_key_padding_mask=None):
        B, T = tgt_ids.shape
        device = tgt_ids.device

        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_emb(tgt_ids) * math.sqrt(self.d_model) + self.pos_emb(pos)

        # Use custom proof self-attn mask if provided (already includes causal masking),
        # otherwise fall back to standard causal mask.
        if proof_self_attn_mask is not None:
            tgt_mask = proof_self_attn_mask
        else:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        # Project encoder output to decoder dim
        enc = self.encoder_proj(encoder_out)

        # Invert attention mask for key_padding_mask (True = ignore)
        # raw_memory_key_padding_mask bypasses inversion (already bool, True=ignore)
        if raw_memory_key_padding_mask is not None:
            memory_key_padding_mask = raw_memory_key_padding_mask
        elif encoder_attention_mask is not None:
            memory_key_padding_mask = encoder_attention_mask == 0
        else:
            memory_key_padding_mask = None

        for i, layer in enumerate(self.layers):
            # Proof layers use premises-only cross-attn mask if provided
            if (proof_cross_attn_mask is not None and proof_layer_start is not None
                    and i >= proof_layer_start):
                cross_mask = proof_cross_attn_mask
            else:
                cross_mask = memory_key_padding_mask
            x = layer(x, enc, tgt_mask=tgt_mask, memory_key_padding_mask=cross_mask)

        # Collect cross-attention weights from all layers
        self.cross_attn_weights = [layer.last_cross_attn_weights for layer in self.layers]

        x = self.norm(x)
        self.last_hidden = x  # (B, T, d_model) — used by answer cls head in trainer
        return self.lm_head(x)  # (B, T, vocab_size)
