import torch
from torch.nn.utils.rnn import pad_sequence


class FOLCollatorV2:
    """Collator for two-decoder architecture (FOLDatasetV2)."""

    def __init__(self, pad_token_id: int):
        self.pad_id = pad_token_id

    def __call__(self, batch):
        input_ids = pad_sequence(
            [x["input_ids"] for x in batch], batch_first=True, padding_value=self.pad_id
        )
        attention_mask = pad_sequence(
            [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
        )
        trans_decoder_input_ids = pad_sequence(
            [x["trans_decoder_input_ids"] for x in batch], batch_first=True, padding_value=self.pad_id
        )
        trans_labels = pad_sequence(
            [x["trans_labels"] for x in batch], batch_first=True, padding_value=-100
        )
        proof_decoder_input_ids = pad_sequence(
            [x["proof_decoder_input_ids"] for x in batch], batch_first=True, padding_value=self.pad_id
        )
        proof_labels = pad_sequence(
            [x["proof_labels"] for x in batch], batch_first=True, padding_value=-100
        )
        # Padding mask for translation hidden states (used by proof decoder cross-attn)
        # 1 = real token, 0 = padding
        trans_padding_mask = (trans_decoder_input_ids != self.pad_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "trans_decoder_input_ids": trans_decoder_input_ids,
            "trans_labels": trans_labels,
            "proof_decoder_input_ids": proof_decoder_input_ids,
            "proof_labels": proof_labels,
            "trans_padding_mask": trans_padding_mask,
        }


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
