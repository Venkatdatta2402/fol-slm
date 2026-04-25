"""Runtime pipeline: NL premises + question → FOL (translation decoder) → clingo → answer + proof.

Usage:
    from scripts.pipeline import load_model, run

    load_model("outputs/v12_translation_4L512/checkpoint_final.pt")
    result = run(
        nl_premises="Anne is kind. If someone is kind they are furry.",
        nl_question="Anne is furry.",
    )
    print(result["answer"])   # True
    for step in result["proof"]:
        print(step)
"""

import sys
import torch
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer
from src.model import FOLModelV2
from scripts.reason import reason

_model     = None
_tokenizer = None
_cfg       = None
_device    = None


def load_model(checkpoint_path: str, config_path: str = "configs/v12_translation.yaml"):
    """Load translation decoder from checkpoint. Safe to call multiple times (cached)."""
    global _model, _tokenizer, _cfg, _device

    with open(config_path) as f:
        _cfg = yaml.safe_load(f)

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(_cfg["model"]["encoder_name"])
    _model     = FOLModelV2(_cfg["model"], vocab_size=_tokenizer.vocab_size).to(_device)

    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    _model.load_state_dict(ckpt["model_state"], strict=False)
    _model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}  (step {ckpt.get('step', '?')})  device={_device}")


def translate(nl_premises: str, nl_question: str) -> str:
    """Run translation decoder: NL → raw FOL string (with <extra_id_N> sentinels)."""
    if _model is None:
        raise RuntimeError("Call load_model() first.")

    extra_id_1 = _tokenizer.convert_tokens_to_ids("<extra_id_1>")
    extra_id_3 = _tokenizer.convert_tokens_to_ids("<extra_id_3>")

    # Two separate text boxes joined by <extra_id_0> sentinel
    input_text = f"{nl_premises.strip()} <extra_id_0> {nl_question.strip()}"
    enc = _tokenizer(
        input_text,
        return_tensors="pt",
        max_length=_cfg["data"]["max_input_len"],
        truncation=True,
    ).to(_device)

    decoder_ids = torch.tensor([[extra_id_1]], device=_device)
    max_len = _cfg["data"]["max_target_len"]

    with torch.no_grad():
        encoder_out = _model.encoder(enc["input_ids"], enc["attention_mask"])
        for _ in range(max_len):
            logits = _model.translation_decoder(decoder_ids, encoder_out, enc["attention_mask"])
            next_token = logits[0, -1].argmax(-1).item()
            decoder_ids = torch.cat(
                [decoder_ids, torch.tensor([[next_token]], device=_device)], dim=1
            )
            if next_token == extra_id_3 or next_token == _tokenizer.eos_token_id:
                break

    return _tokenizer.decode(decoder_ids[0], skip_special_tokens=False)


def run(nl_premises: str, nl_question: str) -> dict:
    """Full pipeline: NL → FOL → clingo → answer + proof.

    Returns:
        {
          "fol_premises": str,
          "fol_question": str,
          "answer":       "True" | "False" | "Unknown",
          "proof":        list[str],
        }
    """
    fol_output = translate(nl_premises, nl_question)
    result     = reason(fol_output)
    return {
        "fol_premises": result["premises_fol"],
        "fol_question": result["question_fol"],
        "answer":       result["answer"],
        "proof":        result["proof"],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/v12_translation_4L512/checkpoint_final.pt")
    parser.add_argument("--config",     default="configs/v12_translation.yaml")
    parser.add_argument("--premises",   required=True)
    parser.add_argument("--question",   required=True)
    args = parser.parse_args()

    load_model(args.checkpoint, args.config)
    out = run(args.premises, args.question)

    print(f"\nFOL Premises:\n{out['fol_premises']}")
    print(f"\nFOL Question: {out['fol_question']}")
    print(f"\nAnswer: {out['answer']}")
    print("\nProof:")
    for step in out["proof"]:
        print(f"  {step}")
