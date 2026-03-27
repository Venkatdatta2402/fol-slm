import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer


class FrozenT5Encoder(nn.Module):
    def __init__(self, model_name: str = "t5-base"):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.d_model

        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.last_hidden_state  # (B, S, d_model)
