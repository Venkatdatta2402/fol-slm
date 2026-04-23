import torch


def cross_attention_entropy(
    attn_weights_list: list[torch.Tensor],
    fol_mask: torch.Tensor | None = None,
) -> float:
    """Compute mean entropy of cross-attention distributions across layers.

    Only positions within the PREMISES_FOL section are included when fol_mask
    is provided — PROOF and ANSWER positions are excluded because those decoder
    steps should use self-attention over already-generated tokens, not the encoder.

    Args:
        attn_weights_list: List of tensors, one per layer.
            Shape per tensor: (B, n_heads, T_dec, T_enc) or (B, T_dec, T_enc).
        fol_mask: Optional float tensor of shape (B, T_dec) with 1.0 at
            PREMISES_FOL positions and 0.0 elsewhere. If None, all positions
            are included (unweighted, original behaviour).

    Returns:
        Scalar mean entropy (nats).
    """
    total_entropy = 0.0
    count = 0
    for w in attn_weights_list:
        # Cast to float32 and renormalize to handle bf16 precision artifacts
        w = w.float().clamp(min=0)
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        w = w.clamp(min=1e-9)
        # entropy shape: (B, n_heads, T_dec) or (B, T_dec)
        entropy = -(w * w.log()).sum(dim=-1)

        if fol_mask is not None:
            # entropy: (B, n_heads, T_dec) — average over heads first
            if entropy.dim() == 3:
                entropy = entropy.mean(dim=1)  # (B, T_dec)
            # Weighted mean: only PREMISES_FOL positions contribute
            masked = (entropy * fol_mask).sum()
            n_valid = fol_mask.sum().clamp(min=1)
            total_entropy += (masked / n_valid).item()
        else:
            total_entropy += entropy.mean().item()

        count += 1
    return total_entropy / count if count > 0 else 0.0


def build_proof_self_attn_mask(
    decoder_input_ids: torch.Tensor,
    extra_id_2_id: int,
    extra_id_3_id: int,
    n_heads: int,
) -> torch.Tensor:
    """Build a per-sample self-attention mask for proof generation.

    At proof positions (after <extra_id_3>), blocks self-attention to the
    FOL premises section (before <extra_id_2>). Proof steps can only attend to:
      - FOL question tokens  (<extra_id_2> .. <extra_id_3>)
      - Previously generated proof steps  (<extra_id_3> .. t-1)

    This forces proof reasoning to be driven by what is being asked (FOL question)
    and what has been derived so far — not by pattern-matching over FOL premises.

    Standard causal masking is included in the returned mask.

    Args:
        decoder_input_ids: (B, T) long tensor.
        extra_id_2_id: Token ID of <extra_id_2> (question sentinel).
        extra_id_3_id: Token ID of <extra_id_3> (proof sentinel).
        n_heads: Number of attention heads.

    Returns:
        Float additive mask of shape (B*n_heads, T, T).
        0.0 = allowed, -inf = blocked.
    """
    B, T = decoder_input_ids.shape
    device = decoder_input_ids.device

    # Start with standard causal mask (upper triangle = -inf)
    causal = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
    mask = causal.unsqueeze(0).expand(B, T, T).clone()  # (B, T, T)

    for b in range(B):
        id2_matches = (decoder_input_ids[b] == extra_id_2_id).nonzero(as_tuple=True)[0]
        id3_matches = (decoder_input_ids[b] == extra_id_3_id).nonzero(as_tuple=True)[0]
        if len(id2_matches) == 0 or len(id3_matches) == 0:
            continue
        id2 = id2_matches[0].item()
        id3 = id3_matches[0].item()
        # Proof positions: rows after <extra_id_3>; block cols before <extra_id_2>
        if id3 + 1 < T and id2 > 0:
            mask[b, id3 + 1:, :id2] = float('-inf')

    # Expand for heads: (B, T, T) -> (B*n_heads, T, T)
    return mask.unsqueeze(1).expand(B, n_heads, T, T).reshape(B * n_heads, T, T)


def build_premises_cross_attn_mask(
    input_ids: torch.Tensor,
    extra_id_0_id: int,
) -> torch.Tensor:
    """Build cross-attention key_padding_mask restricting proof layers to premises only.

    Finds <extra_id_0> in each encoder input sequence. Positions at and after
    <extra_id_0> (the NL question) are masked out — proof layers can only
    cross-attend to NL premises positions.

    Args:
        input_ids: (B, S_enc) encoder input token IDs.
        extra_id_0_id: Token ID of <extra_id_0> (premise/question separator).

    Returns:
        Bool tensor of shape (B, S_enc). True = ignore, False = attend.
    """
    B, S = input_ids.shape
    device = input_ids.device
    mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    for b in range(B):
        matches = (input_ids[b] == extra_id_0_id).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            pos = matches[0].item()
            mask[b, pos:] = True  # ignore question and padding
    return mask


def build_fol_mask(
    decoder_input_ids: torch.Tensor,
    proof_sentinel_id: int,
) -> torch.Tensor:
    """Build a float mask that is 1.0 for PREMISES_FOL positions only.

    Positions from <extra_id_3> (PROOF sentinel) onward get 0.0.
    Positions before it (the FOL premises lines) get 1.0.

    Args:
        decoder_input_ids: (B, T_dec) long tensor.
        proof_sentinel_id: Token ID of <extra_id_3>.

    Returns:
        Float mask of shape (B, T_dec).
    """
    B, T = decoder_input_ids.shape
    mask = torch.ones(B, T, dtype=torch.float32, device=decoder_input_ids.device)
    for b in range(B):
        positions = (decoder_input_ids[b] == proof_sentinel_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            # Zero out from the first <extra_id_2> token onward
            mask[b, positions[0]:] = 0.0
    return mask
