import torch
import torch.nn as nn
from .encoder import FrozenT5Encoder
from .decoder import TransformerDecoder


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

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        encoder_out = self.encoder(input_ids, attention_mask)
        logits = self.decoder(decoder_input_ids, encoder_out, attention_mask)
        return logits  # (B, T, vocab_size)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())
