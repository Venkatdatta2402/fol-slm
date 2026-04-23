"""Inference script for generating outputs from a trained FOL SLM model.

Loads a checkpoint and runs autoregressive decoding on input premises.

Usage:
    python scripts/generate.py --checkpoint outputs/final/checkpoint_final.pt \
        --input "All cats are animals. Tom is a cat."
    python scripts/generate.py --checkpoint outputs/final/checkpoint_final.pt \
        --input-file data/processed/dev.jsonl
"""

import sys
import json
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModel
from src.utils.attention import build_proof_self_attn_mask


def generate(
    model: FOLModel,
    tokenizer,
    input_text: str,
    device: torch.device,
    max_len: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
    max_input_len: int = 512,
) -> str:
    """Autoregressive decoding from premises.

    Args:
        model: Trained FOLModel.
        tokenizer: Tokenizer matching the model.
        input_text: Natural language premises.
        device: Torch device.
        max_len: Maximum generation length.
        temperature: Sampling temperature (1.0 = no change, <1.0 = sharper).
        top_k: If > 0, sample from top-k tokens only.

    Returns:
        Generated text string.
    """
    model.eval()

    enc = tokenizer(
        input_text,
        max_length=max_input_len,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        encoder_out = model.encoder(input_ids, attention_mask)

    # Start with <extra_id_1> to match training distribution.
    # During training, decoder_input_ids = target_ids[:-1] which starts with <extra_id_1>.
    # Starting from [pad] at inference is out-of-distribution and produces garbage.
    extra_id_1 = tokenizer.convert_tokens_to_ids("<extra_id_1>")
    extra_id_2 = tokenizer.convert_tokens_to_ids("<extra_id_2>")
    extra_id_3 = tokenizer.convert_tokens_to_ids("<extra_id_3>")
    extra_id_4 = tokenizer.convert_tokens_to_ids("<extra_id_4>")
    n_heads = model.decoder.layers[0].self_attn.num_heads
    decoder_ids = torch.tensor([[extra_id_1]], device=device)

    with torch.no_grad():
        for _ in range(max_len):
            proof_mask = build_proof_self_attn_mask(decoder_ids, extra_id_2, extra_id_3, n_heads)
            logits = model.decoder(decoder_ids, encoder_out, attention_mask, proof_mask)
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_vals, top_idx = torch.topk(next_logits, top_k, dim=-1)
                mask = torch.full_like(next_logits, float("-inf"))
                mask.scatter_(1, top_idx, top_vals)
                next_logits = mask

            if temperature == 1.0 and top_k == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            decoder_ids = torch.cat([decoder_ids, next_token], dim=1)

            # Stop at <extra_id_4> (end of proof) or EOS — answer comes from cls head
            if next_token.item() in (extra_id_4, tokenizer.eos_token_id):
                break

    output_ids = decoder_ids[0, 1:]  # skip the <extra_id_1> start token
    # Keep special tokens so section sentinels (<extra_id_2>, <extra_id_3>, <extra_id_4>) are visible.
    # Prepend <extra_id_1> so parsers can find the PREMISES_FOL section boundary.
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    text = text.replace(tokenizer.eos_token, "").strip()
    return "<extra_id_1> " + text


def main(
    checkpoint_path: str,
    config_path: str = "configs/final.yaml",
    input_text: str | None = None,
    input_file: str | None = None,
    max_len: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    model = FOLModel(cfg["model"], vocab_size=tokenizer.vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Device: {device}\n")

    inputs = []
    if input_text:
        inputs.append(input_text)
    elif input_file:
        with open(input_file) as f:
            for line in f:
                sample = json.loads(line)
                inputs.append(sample["premises"])
    else:
        print("Error: Provide --input or --input-file")
        sys.exit(1)

    max_input_len = cfg["data"]["max_input_len"]
    for i, text in enumerate(inputs):
        raw = generate(model, tokenizer, text, device, max_len, temperature, top_k, max_input_len)
        # Strip FOL reasoning chain — return only the NL answer after <extra_id_4>
        if "<extra_id_4>" in raw:
            answer = raw.split("<extra_id_4>")[1].strip()
        else:
            answer = raw  # fallback: model didn't emit answer sentinel yet
        print(f"[{i + 1}] Input:  {text}")
        print(f"     Answer: {answer}")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--input", dest="input_text", help="Single input premise")
    parser.add_argument("--input-file", help="JSONL file with premises")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.input_text, args.input_file,
         args.max_len, args.temperature, args.top_k)
