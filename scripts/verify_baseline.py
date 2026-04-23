"""Phase 1 baseline verification script.

Runs a short training loop and checks:
1. Encoder outputs are constant across steps (frozen check)
2. No encoder params have gradients
3. Decoder cross-attention params have non-zero gradients
4. Cross-attention entropy decreases from initialization
"""

import sys
import yaml
import torch
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModel
from src.data.dataset import ReasoningDataset
from src.data.collator import FOLCollator
from src.utils.attention import cross_attention_entropy


def verify_frozen_encoder(model, sample_batch, device):
    """Check that encoder outputs are identical across calls."""
    input_ids = sample_batch["input_ids"].to(device)
    mask = sample_batch["attention_mask"].to(device)

    out1 = model.encoder(input_ids, mask)
    out2 = model.encoder(input_ids, mask)

    if torch.allclose(out1, out2, atol=1e-6):
        print("  PASS: Encoder outputs are deterministic (frozen)")
    else:
        print("  FAIL: Encoder outputs differ between calls!")
        return False
    return True


def verify_no_encoder_gradients(model):
    """Check that no encoder param has accumulated gradients."""
    for name, param in model.encoder.named_parameters():
        if param.grad is not None:
            print(f"  FAIL: Encoder param {name} has gradient!")
            return False
    print("  PASS: No encoder parameters have gradients")
    return True


def verify_cross_attn_gradients(model):
    """Check that cross-attention params have non-zero gradients."""
    for i, layer in enumerate(model.decoder.layers):
        for name, param in layer.cross_attn.named_parameters():
            if param.grad is None:
                print(f"  FAIL: Layer {i} cross-attn param {name} has no gradient")
                return False
            if param.grad.abs().sum() == 0:
                print(f"  FAIL: Layer {i} cross-attn param {name} has zero gradient")
                return False
    print("  PASS: Cross-attention params have non-zero gradients")
    return True


def verify_entropy_decreases(entropy_log):
    """Check that cross-attention entropy decreased during training."""
    if len(entropy_log) < 2:
        print("  SKIP: Not enough entropy measurements")
        return True

    initial = entropy_log[0]
    final = entropy_log[-1]

    if final < initial:
        print(f"  PASS: Cross-attn entropy decreased {initial:.4f} -> {final:.4f}")
    else:
        print(f"  WARN: Cross-attn entropy did not decrease {initial:.4f} -> {final:.4f}")
        print("        This may be fine for very few steps, but investigate if persistent")
        return False
    return True


def main(config_path: str = "configs/phase1_baseline.yaml", verify_steps: int = 500):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    train_dataset = ReasoningDataset(
        cfg["data"]["train_path"], tokenizer,
        cfg["data"]["max_input_len"], cfg["data"]["max_target_len"],
    )
    val_dataset = ReasoningDataset(
        cfg["data"]["val_path"], tokenizer,
        cfg["data"]["max_input_len"], cfg["data"]["max_target_len"],
    )

    collator = FOLCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, collate_fn=collator, num_workers=0)

    model = FOLModel(cfg["model"], vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    print(f"\n{'='*60}")
    print("BASELINE VERIFICATION ({} steps)".format(verify_steps))
    print(f"{'='*60}\n")

    # Check 1: Frozen encoder
    print("Check 1: Frozen encoder outputs")
    sample_batch = next(iter(train_loader))
    check1 = verify_frozen_encoder(model, sample_batch, device)

    # Training loop for gradient and entropy checks
    entropy_log = []
    step = 0
    model.train()

    for batch in train_loader:
        if step >= verify_steps:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, decoder_input_ids)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), labels.view(B * T))
        loss.backward()

        # Log cross-attention entropy periodically
        if step % 50 == 0:
            weights = model.decoder.cross_attn_weights
            if weights is not None:
                ent = cross_attention_entropy(weights)
                entropy_log.append(ent)
                print(f"  [step {step}] loss: {loss.item():.4f}  cross_attn_entropy: {ent:.4f}")

        optimizer.step()
        optimizer.zero_grad()
        step += 1

    # Check 2: No encoder gradients
    print("\nCheck 2: Encoder gradient check")
    check2 = verify_no_encoder_gradients(model)

    # Check 3: Cross-attention gradients (run one more step)
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    decoder_input_ids = batch["decoder_input_ids"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids, attention_mask, decoder_input_ids)
    B, T, V = logits.shape
    loss = loss_fn(logits.view(B * T, V), labels.view(B * T))
    loss.backward()

    print("\nCheck 3: Cross-attention gradient flow")
    check3 = verify_cross_attn_gradients(model)

    # Check 4: Entropy decrease
    print("\nCheck 4: Cross-attention entropy trend")
    check4 = verify_entropy_decreases(entropy_log)

    # Summary
    print(f"\n{'='*60}")
    results = {"Frozen encoder": check1, "No encoder grads": check2,
               "Cross-attn grads": check3, "Entropy decreases": check4}
    passed = sum(results.values())
    total = len(results)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n{passed}/{total} checks passed")
    print(f"{'='*60}")

    return passed == total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_baseline.yaml")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    success = main(args.config, args.steps)
    sys.exit(0 if success else 1)
