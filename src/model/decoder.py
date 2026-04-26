import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
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

    # ── KV cache helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _proj_kv(x: torch.Tensor, mha: nn.MultiheadAttention):
        """Extract K and V projections from x using MHA's internal weights."""
        d = mha.embed_dim
        W, b = mha.in_proj_weight, mha.in_proj_bias
        k = F.linear(x, W[d:2*d],  b[d:2*d]  if b is not None else None)
        v = F.linear(x, W[2*d:3*d], b[2*d:3*d] if b is not None else None)
        return k, v

    @staticmethod
    def _attn_with_kv(q_input: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      mha: nn.MultiheadAttention,
                      key_padding_mask: torch.Tensor = None):
        """Compute MHA given pre-projected K, V and raw Q input."""
        d        = mha.embed_dim
        n_heads  = mha.num_heads
        head_dim = d // n_heads
        B, T_q   = q_input.shape[:2]
        T_kv     = k.shape[1]

        W, b = mha.in_proj_weight, mha.in_proj_bias
        q = F.linear(q_input, W[:d], b[:d] if b is not None else None)

        q = q.view(B, T_q,  n_heads, head_dim).transpose(1, 2)  # (B, H, T_q,  head_dim)
        k = k.view(B, T_kv, n_heads, head_dim).transpose(1, 2)  # (B, H, T_kv, head_dim)
        v = v.view(B, T_kv, n_heads, head_dim).transpose(1, 2)  # (B, H, T_kv, head_dim)

        attn_bias = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, T_kv) bool, True = ignore
            attn_bias = torch.zeros(B, n_heads, T_q, T_kv,
                                    device=q_input.device, dtype=q_input.dtype)
            attn_bias.masked_fill_(key_padding_mask[:, None, None, :], float('-inf'))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).contiguous().view(B, T_q, d)
        return F.linear(out, mha.out_proj.weight, mha.out_proj.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x, encoder_out, tgt_mask=None, memory_key_padding_mask=None,
                past_kv=None, use_cache=False):
        """
        past_kv: None  or  (self_k, self_v, cross_k, cross_v)
        Returns: x  [, new_kv]   — new_kv only when use_cache=True
        """

        if use_cache and past_kv is not None:
            # ── Incremental decode: x is the new token only (B, 1, d) ──────
            self_k_cache, self_v_cache, cross_k, cross_v = past_kv

            # Self-attention: project new token, append to cache
            k_new, v_new = self._proj_kv(x, self.self_attn)
            self_k = torch.cat([self_k_cache, k_new], dim=1)
            self_v = torch.cat([self_v_cache, v_new], dim=1)
            attn_out = self._attn_with_kv(x, self_k, self_v, self.self_attn)
            x = self.norm1(x + self.dropout(attn_out))

            # Cross-attention: reuse cached encoder K, V
            attn_out = self._attn_with_kv(x, cross_k, cross_v, self.cross_attn,
                                           key_padding_mask=memory_key_padding_mask)
            self.last_cross_attn_weights = None
            x = self.norm2(x + self.dropout(attn_out))

            x = self.norm3(x + self.dropout(self.ff(x)))
            return x, (self_k, self_v, cross_k, cross_v)

        else:
            # ── Full-sequence forward (training / first decode step) ─────────
            if use_cache:
                # Pre-compute K, V projections before the forward passes
                self_k, self_v = self._proj_kv(x, self.self_attn)
                cross_k, cross_v = self._proj_kv(encoder_out, self.cross_attn)

            attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
            x = self.norm1(x + self.dropout(attn_out))

            attn_out, cross_w = self.cross_attn(
                x, encoder_out, encoder_out,
                key_padding_mask=memory_key_padding_mask,
                average_attn_weights=False,
            )
            self.last_cross_attn_weights = cross_w
            x = self.norm2(x + self.dropout(attn_out))

            x = self.norm3(x + self.dropout(self.ff(x)))

            if use_cache:
                return x, (self_k, self_v, cross_k, cross_v)
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
        encoder_d_model: int = 768,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_emb  = nn.Embedding(vocab_size, d_model)
        self.pos_emb    = nn.Embedding(max_seq_len, d_model)

        self.encoder_proj = (
            nn.Linear(encoder_d_model, d_model)
            if encoder_d_model != d_model else nn.Identity()
        )

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.cross_attn_weights = None
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt_ids, encoder_out, encoder_attention_mask=None,
                proof_self_attn_mask=None, proof_cross_attn_mask=None, proof_layer_start=None,
                raw_memory_key_padding_mask=None,
                past_key_values=None, use_cache=False):
        """
        past_key_values: None or list of per-layer (self_k, self_v, cross_k, cross_v)
        use_cache: if True, returns (logits, new_past_key_values)
        """
        B, T = tgt_ids.shape
        device = tgt_ids.device

        # Offset positional embeddings when using KV cache (incremental decode)
        past_len = past_key_values[0][0].shape[1] if past_key_values is not None else 0
        pos = torch.arange(past_len, past_len + T, device=device).unsqueeze(0)
        x   = self.token_emb(tgt_ids) * math.sqrt(self.d_model) + self.pos_emb(pos)

        if proof_self_attn_mask is not None:
            tgt_mask = proof_self_attn_mask
        else:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        enc = self.encoder_proj(encoder_out)

        if raw_memory_key_padding_mask is not None:
            memory_key_padding_mask = raw_memory_key_padding_mask
        elif encoder_attention_mask is not None:
            memory_key_padding_mask = encoder_attention_mask == 0
        else:
            memory_key_padding_mask = None

        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            cross_mask = memory_key_padding_mask
            if (proof_cross_attn_mask is not None and proof_layer_start is not None
                    and i >= proof_layer_start):
                cross_mask = proof_cross_attn_mask

            past_kv_i = past_key_values[i] if past_key_values is not None else None

            if use_cache:
                x, new_kv = layer(x, enc, tgt_mask=tgt_mask,
                                  memory_key_padding_mask=cross_mask,
                                  past_kv=past_kv_i, use_cache=True)
                new_past_key_values.append(new_kv)
            else:
                x = layer(x, enc, tgt_mask=tgt_mask,
                          memory_key_padding_mask=cross_mask)

        self.cross_attn_weights = [layer.last_cross_attn_weights for layer in self.layers]
        x = self.norm(x)
        self.last_hidden = x
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_past_key_values
        return logits
