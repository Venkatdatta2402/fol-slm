import torch
import torch.nn as nn
from .encoder import FrozenT5Encoder
from .decoder import TransformerDecoder


class FOLModelV2(nn.Module):
    """Two-decoder architecture:
      - translation_decoder: NL → FOL premises + FOL question (stops at <extra_id_3>)
      - proof_decoder: FOL representations → proof chain (stops at <extra_id_4>)
      - answer_cls_head: proof representation → True/False/Unknown

    Trained sequentially:
      Phase 1: train translation_decoder only (proof_decoder frozen/unused)
      Phase 2: freeze translation_decoder, train proof_decoder
      Phase 3: freeze both decoders, train cls_head
    """

    def __init__(self, config: dict, vocab_size: int):
        super().__init__()
        self.encoder = FrozenT5Encoder(config["encoder_name"])

        trans_cfg = config["translation_decoder"]
        self.translation_decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=trans_cfg["dim"],
            n_heads=trans_cfg["heads"],
            n_layers=trans_cfg["layers"],
            d_ff=trans_cfg["ff_dim"],
            dropout=trans_cfg.get("dropout", 0.1),
            max_seq_len=config["max_seq_len"],
            encoder_d_model=self.encoder.hidden_size,
        )

        proof_cfg = config["proof_decoder"]
        self.proof_decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=proof_cfg["dim"],
            n_heads=proof_cfg["heads"],
            n_layers=proof_cfg["layers"],
            d_ff=proof_cfg["ff_dim"],
            dropout=proof_cfg.get("dropout", 0.35),
            max_seq_len=config["max_seq_len"],
            encoder_d_model=trans_cfg["dim"],  # cross-attends to translation hidden states
        )

        # Cls head reads proof_decoder hidden state at last proof position
        self.answer_cls_head = nn.Linear(proof_cfg["dim"], 3)

    def forward_translation(self, input_ids, attention_mask, trans_decoder_input_ids):
        """Run encoder + translation decoder. Returns (logits, hidden_states)."""
        encoder_out = self.encoder(input_ids, attention_mask)
        trans_logits = self.translation_decoder(trans_decoder_input_ids, encoder_out, attention_mask)
        trans_hidden = self.translation_decoder.last_hidden  # (B, T_trans, trans_dim)
        return trans_logits, trans_hidden, encoder_out

    def forward_proof(self, trans_hidden, proof_decoder_input_ids, trans_padding_mask=None):
        """Run proof decoder cross-attending to translation hidden states."""
        proof_logits = self.proof_decoder(proof_decoder_input_ids, trans_hidden, trans_padding_mask)
        return proof_logits

    def forward(self, input_ids, attention_mask, trans_decoder_input_ids,
                proof_decoder_input_ids, trans_padding_mask=None):
        trans_logits, trans_hidden, _ = self.forward_translation(
            input_ids, attention_mask, trans_decoder_input_ids
        )
        proof_logits = self.forward_proof(trans_hidden, proof_decoder_input_ids, trans_padding_mask)
        return trans_logits, proof_logits

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())


class FOLModel(nn.Module):
    def __init__(self, config: dict, vocab_size: int):
        super().__init__()
        self.encoder = FrozenT5Encoder(config["encoder_name"])
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=config["decoder_dim"],
            n_heads=config["decoder_heads"],
            n_layers=config["decoder_layers"],
            d_ff=config["decoder_ff_dim"],
            dropout=config["decoder_dropout"],
            max_seq_len=config["max_seq_len"],
            encoder_d_model=self.encoder.hidden_size,
        )
        # Classification head: activated at <extra_id_4> position during training.
        # Takes decoder hidden state → 3 classes (True=0, False=1, Unknown=2).
        # Training-only auxiliary loss; ignored at inference.
        self.answer_cls_head = nn.Linear(config["decoder_dim"], 3)

    def forward(self, input_ids, attention_mask, decoder_input_ids, proof_self_attn_mask=None):
        encoder_out = self.encoder(input_ids, attention_mask)
        logits = self.decoder(decoder_input_ids, encoder_out, attention_mask, proof_self_attn_mask)
        return logits  # (B, T, vocab_size)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())
