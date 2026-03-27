import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class FOLIODataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_input_len: int = 256,
        max_target_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.samples = self._load(data_path)

    def _load(self, path: str):
        samples = []
        with open(path) as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Expects fields: "premises" (str), "logic" (str target sequence)
        source = sample["premises"]
        target = sample["logic"]

        enc = self.tokenizer(
            source,
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        dec = self.tokenizer(
            target,
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        target_ids = dec["input_ids"].squeeze(0)

        # Decoder input: shift right (teacher forcing)
        decoder_input_ids = target_ids[:-1]
        labels = target_ids[1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
