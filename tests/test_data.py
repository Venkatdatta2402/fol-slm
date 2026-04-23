import sys
import json
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import ReasoningDataset
from src.data.collator import FOLCollator


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("t5-small")


@pytest.fixture
def sample_data(tmp_path):
    data = [
        {"premises": "All cats are animals. Tom is a cat.", "logic": "ForAll(x, Cat(x) -> Animal(x)). Cat(Tom). Therefore Animal(Tom). Tom is an animal."},
        {"premises": "No birds can swim. Tweety is a bird.", "logic": "ForAll(x, Bird(x) -> Not(Swim(x))). Bird(Tweety). Therefore Not(Swim(Tweety)). Tweety cannot swim."},
        {"premises": "Some dogs are friendly.", "logic": "Exists(x, Dog(x) & Friendly(x)). There exists a friendly dog."},
    ]
    path = tmp_path / "test.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path


class TestReasoningDataset:
    def test_load_and_length(self, sample_data, tokenizer):
        dataset = ReasoningDataset(str(sample_data), tokenizer, max_input_len=64, max_target_len=64)
        assert len(dataset) == 3

    def test_sample_keys(self, sample_data, tokenizer):
        dataset = ReasoningDataset(str(sample_data), tokenizer, max_input_len=64, max_target_len=64)
        sample = dataset[0]
        assert set(sample.keys()) == {"input_ids", "attention_mask", "decoder_input_ids", "labels"}

    def test_teacher_forcing_shift(self, sample_data, tokenizer):
        dataset = ReasoningDataset(str(sample_data), tokenizer, max_input_len=64, max_target_len=64)
        sample = dataset[0]
        # decoder_input_ids should be target[:-1], labels should be target[1:]
        # So decoder_input_ids[1:] should equal labels[:-1]
        assert torch.equal(sample["decoder_input_ids"][1:], sample["labels"][:-1])
        # lengths should be equal
        assert len(sample["decoder_input_ids"]) == len(sample["labels"])

    def test_tensors_are_1d(self, sample_data, tokenizer):
        dataset = ReasoningDataset(str(sample_data), tokenizer, max_input_len=64, max_target_len=64)
        sample = dataset[0]
        for key in sample:
            assert sample[key].dim() == 1, f"{key} should be 1D"


class TestFOLCollator:
    def test_padding_shapes(self, sample_data, tokenizer):
        dataset = ReasoningDataset(str(sample_data), tokenizer, max_input_len=64, max_target_len=64)
        collator = FOLCollator(tokenizer.pad_token_id)
        batch = collator([dataset[0], dataset[1]])

        assert batch["input_ids"].dim() == 2
        assert batch["labels"].dim() == 2
        B = 2
        assert batch["input_ids"].shape[0] == B
        assert batch["labels"].shape[0] == B

    def test_labels_padded_with_neg100(self, sample_data, tokenizer):
        dataset = ReasoningDataset(str(sample_data), tokenizer, max_input_len=64, max_target_len=64)
        collator = FOLCollator(tokenizer.pad_token_id)

        # Create batch with different length targets to force padding
        batch = collator([dataset[0], dataset[2]])
        labels = batch["labels"]

        # At least one sample should have -100 padding (if different lengths)
        if labels.shape[1] > 1:
            # Check that padding positions use -100
            has_padding = (labels == -100).any()
            # This should be true if the two samples have different target lengths
            # (which they likely do given different content)
            assert has_padding or labels.shape[1] == min(
                len(dataset[0]["labels"]), len(dataset[2]["labels"])
            )

    def test_attention_mask_padded_with_zero(self, sample_data, tokenizer):
        dataset = ReasoningDataset(str(sample_data), tokenizer, max_input_len=64, max_target_len=64)
        collator = FOLCollator(tokenizer.pad_token_id)
        batch = collator([dataset[0], dataset[1]])
        mask = batch["attention_mask"]
        # Should only contain 0s and 1s
        assert set(mask.unique().tolist()).issubset({0, 1})
