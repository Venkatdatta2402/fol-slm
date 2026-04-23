import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

ANSWER_TO_IDX = {"True": 0, "False": 1, "Unknown": 2}


class FOLDatasetV2(Dataset):
    """Dataset for two-decoder architecture.

    Returns separate inputs for translation decoder and proof decoder:
      - trans_decoder_input_ids: target up to (not incl.) <extra_id_3>
      - trans_labels: shifted, last label = <extra_id_3>
      - proof_decoder_input_ids: from <extra_id_3> up to (not incl.) <extra_id_4>
      - proof_labels: shifted, last label = <extra_id_4>

    Neither decoder sees the answer token in its input or labels.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_input_len: int = 512,
        max_target_len: int = 512,
        max_qdep: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.extra_id_3 = tokenizer.convert_tokens_to_ids("<extra_id_3>")
        self.extra_id_4 = tokenizer.convert_tokens_to_ids("<extra_id_4>")
        self.samples = self._load(data_path, max_qdep)

    def _load(self, path, max_qdep):
        samples = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                if max_qdep is not None and ex.get("qdep", 0) > max_qdep:
                    continue
                samples.append(ex)
        return samples

    def get_sample_weights(self) -> list[float]:
        from collections import Counter

        def _class_key(ex):
            q = ex["premises"].split("<extra_id_0>", 1)[1].strip().lower() \
                if "<extra_id_0>" in ex["premises"] else ""
            is_neg = any(w in q for w in ["not", "n't", "never", "no "])
            return f"{'neg' if is_neg else 'pos'}_{ex['answer']}"

        counts = Counter(_class_key(ex) for ex in self.samples)
        max_count = max(counts.values())
        rare_classes = {"pos_False", "neg_True"}
        return [
            max_count / counts[_class_key(ex)] if _class_key(ex) in rare_classes else 1.0
            for ex in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        enc = self.tokenizer(
            sample["premises"], max_length=self.max_input_len, truncation=True,
            padding=False, return_tensors="pt",
        )
        dec = self.tokenizer(
            sample["logic"], max_length=self.max_target_len, truncation=True,
            padding=False, return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        target_ids = dec["input_ids"].squeeze(0)

        id3_pos = (target_ids == self.extra_id_3).nonzero(as_tuple=True)[0]
        id4_pos = (target_ids == self.extra_id_4).nonzero(as_tuple=True)[0]

        if len(id3_pos) == 0 or len(id4_pos) == 0:
            # Fallback: shouldn't happen with well-formed data
            empty = torch.tensor([self.extra_id_3], dtype=torch.long)
            return {
                "input_ids": input_ids, "attention_mask": attention_mask,
                "trans_decoder_input_ids": target_ids[:-1],
                "trans_labels": target_ids[1:],
                "proof_decoder_input_ids": empty,
                "proof_labels": empty,
            }

        id3 = id3_pos[0].item()
        id4 = id4_pos[0].item()

        # Translation: input = [<extra_id_1> ... FOL_question_tokens]
        # Labels: shifted by 1, last label is <extra_id_3> (model learns to stop here)
        trans_input  = target_ids[:id3]
        trans_labels = target_ids[1:id3 + 1]

        # Proof: input = [<extra_id_3>, proof_step_1, ...]
        # Labels: shifted by 1, last label is <extra_id_4> (model learns to stop here)
        proof_input  = target_ids[id3:id4]
        proof_labels = target_ids[id3 + 1:id4 + 1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "trans_decoder_input_ids": trans_input,
            "trans_labels": trans_labels,
            "proof_decoder_input_ids": proof_input,
            "proof_labels": proof_labels,
        }


class FOLDatasetV3(Dataset):
    """Dataset for V13 two-decoder architecture.

    Proof decoder input starts at <extra_id_2> (FOL question) and includes gold proof tokens.
    No self-attn mask needed — FOL premises are never in the proof decoder input.

    Returns:
      - trans_decoder_input_ids: same as V2 (NL→FOL translation target)
      - trans_labels: same as V2
      - proof_decoder_input_ids: [<extra_id_2>, FOL_question, <extra_id_3>, proof_{0..t-1}]
      - proof_labels: shifted, last label = <extra_id_4>
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_input_len: int = 512,
        max_target_len: int = 512,
        max_qdep: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.extra_id_2 = tokenizer.convert_tokens_to_ids("<extra_id_2>")
        self.extra_id_3 = tokenizer.convert_tokens_to_ids("<extra_id_3>")
        self.extra_id_4 = tokenizer.convert_tokens_to_ids("<extra_id_4>")
        self.samples = self._load(data_path, max_qdep)

    def _load(self, path, max_qdep):
        samples = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                if max_qdep is not None and ex.get("qdep", 0) > max_qdep:
                    continue
                samples.append(ex)
        return samples

    def get_sample_weights(self) -> list[float]:
        from collections import Counter

        def _class_key(ex):
            q = ex["premises"].split("<extra_id_0>", 1)[1].strip().lower() \
                if "<extra_id_0>" in ex["premises"] else ""
            is_neg = any(w in q for w in ["not", "n't", "never", "no "])
            return f"{'neg' if is_neg else 'pos'}_{ex['answer']}"

        counts = Counter(_class_key(ex) for ex in self.samples)
        max_count = max(counts.values())
        rare_classes = {"pos_False", "neg_True"}
        return [
            max_count / counts[_class_key(ex)] if _class_key(ex) in rare_classes else 1.0
            for ex in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        enc = self.tokenizer(
            sample["premises"], max_length=self.max_input_len, truncation=True,
            padding=False, return_tensors="pt",
        )
        dec = self.tokenizer(
            sample["logic"], max_length=self.max_target_len, truncation=True,
            padding=False, return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        target_ids = dec["input_ids"].squeeze(0)

        id2_pos = (target_ids == self.extra_id_2).nonzero(as_tuple=True)[0]
        id3_pos = (target_ids == self.extra_id_3).nonzero(as_tuple=True)[0]
        id4_pos = (target_ids == self.extra_id_4).nonzero(as_tuple=True)[0]

        if len(id2_pos) == 0 or len(id3_pos) == 0 or len(id4_pos) == 0:
            empty = torch.tensor([self.extra_id_3], dtype=torch.long)
            return {
                "input_ids": input_ids, "attention_mask": attention_mask,
                "trans_decoder_input_ids": target_ids[:-1],
                "trans_labels": target_ids[1:],
                "proof_decoder_input_ids": empty,
                "proof_labels": empty,
            }

        id2 = id2_pos[0].item()
        id3 = id3_pos[0].item()
        id4 = id4_pos[0].item()

        # Translation: [<extra_id_1>, FOL_premises, <extra_id_2>, FOL_question]
        # Labels: shifted by 1, last label = <extra_id_3>
        trans_input  = target_ids[:id3]
        trans_labels = target_ids[1:id3 + 1]

        # Proof: [<extra_id_2>, FOL_question, <extra_id_3>, proof_{0..t-1}]
        # Labels: shifted by 1, last label = <extra_id_4>
        proof_input  = target_ids[id2:id4]
        proof_labels = target_ids[id2 + 1:id4 + 1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "trans_decoder_input_ids": trans_input,
            "trans_labels": trans_labels,
            "proof_decoder_input_ids": proof_input,
            "proof_labels": proof_labels,
        }


class ReasoningDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_input_len: int = 256,
        max_target_len: int = 256,
        max_qdep: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.extra_id_4 = tokenizer.convert_tokens_to_ids("<extra_id_4>")
        self.samples = self._load(data_path, max_qdep)

    def _load(self, path: str, max_qdep: int | None):
        samples = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                if max_qdep is not None and ex.get("qdep", 0) > max_qdep:
                    continue
                samples.append(ex)
        return samples

    def get_sample_weights(self) -> list[float]:
        """Boost only underrepresented classes (pos_False, neg_True) without penalising others.

        Weight = max_count / class_count for the two rare classes; 1.0 for everything else.
        This oversamples pos_False and neg_True to match the most common class frequency
        while leaving Unknown and the dominant True/False classes at their natural rate.
        """
        from collections import Counter

        def _class_key(ex):
            q = ex["premises"].split("<extra_id_0>", 1)[1].strip().lower() \
                if "<extra_id_0>" in ex["premises"] else ""
            is_neg = any(w in q for w in ["not", "n't", "never", "no "])
            return f"{'neg' if is_neg else 'pos'}_{ex['answer']}"

        counts = Counter(_class_key(ex) for ex in self.samples)
        max_count = max(counts.values())
        rare_classes = {"pos_False", "neg_True"}
        weights = [
            max_count / counts[_class_key(ex)] if _class_key(ex) in rare_classes else 1.0
            for ex in self.samples
        ]
        return weights

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
        labels = target_ids[1:].clone()

        # Mask answer tokens from labels: decoder is trained on proof only.
        # Keep <extra_id_4> itself as a label (decoder learns to emit it),
        # but mask everything after it (the answer word + EOS) to -100.
        answer_positions = (labels == self.extra_id_4).nonzero(as_tuple=True)[0]
        if len(answer_positions) > 0:
            mask_from = answer_positions[0].item() + 1
            labels[mask_from:] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
