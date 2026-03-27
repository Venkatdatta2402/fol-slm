import torch
from torch.nn.utils.rnn import pad_sequence


class FOLCollator:
    def __init__(self, pad_token_id: int):
        self.pad_id = pad_token_id

    def __call__(self, batch):
        input_ids = pad_sequence(
            [x["input_ids"] for x in batch], batch_first=True, padding_value=self.pad_id
        )
        attention_mask = pad_sequence(
            [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
        )
        decoder_input_ids = pad_sequence(
            [x["decoder_input_ids"] for x in batch], batch_first=True, padding_value=self.pad_id
        )
        labels = pad_sequence(
            [x["labels"] for x in batch], batch_first=True, padding_value=-100  # ignore in loss
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
