import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from src.utils.attention import cross_attention_entropy, build_fol_mask, build_proof_self_attn_mask


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config: dict,
        device: torch.device,
        logging_config: dict | None = None,
        diagnostics_config: dict | None = None,
        proof_sentinel_id: int | None = None,
        curriculum_loaders: list[tuple[int, DataLoader]] | None = None,
        answer_cls_id: int | None = None,
        answer_tok_to_cls: dict | None = None,
        extra_id_2_id: int | None = None,
        extra_id_3_id: int | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.proof_sentinel_id = proof_sentinel_id
        # answer_cls_id: token id of <extra_id_4>; triggers cls head loss at that position
        # answer_tok_to_cls: {token_id: class_idx} mapping True→0, False→1, Unknown→2
        self.answer_cls_id = answer_cls_id
        self.answer_tok_to_cls = answer_tok_to_cls or {}
        self._last_cls_loss: float | None = None
        self._last_decoder_input_ids: torch.Tensor | None = None
        # Token IDs for building proof self-attention mask
        self.extra_id_2_id = extra_id_2_id
        self.extra_id_3_id = extra_id_3_id
        # EMAs for auto-calibrating cls coeff at answer_cls_start_step
        # coeff = target_ratio * ema_ce / ema_cls  (target_ratio = 0.05 → cls contributes 5% of CE)
        self._ema_ce: float | None = None
        self._ema_cls: float | None = None
        self._ema_alpha = 0.05  # smoothing factor; ~20-step window
        # curriculum_loaders: sorted list of (until_step, loader) pairs.
        # At each phase boundary, self.train_loader is replaced with the next loader.
        self._curriculum_loaders = sorted(curriculum_loaders or [], key=lambda x: x[0])
        self.config = config
        self.device = device
        self.logging_config = logging_config or {}
        self.diagnostics_config = diagnostics_config or {}

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01),
        )
        # Convert training steps → optimizer steps (scheduler steps once per grad_accum)
        grad_accum = config["grad_accum_steps"]
        optimizer_steps = config["max_steps"] // grad_accum
        warmup_optimizer_steps = config["warmup_steps"] // grad_accum
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_optimizer_steps,
            num_training_steps=optimizer_steps,
        )
        # Precision: bf16 preferred (same range as fp32, no overflow); fp16 fallback
        self.use_bf16 = config.get("bf16", False)
        self.use_fp16 = config.get("fp16", False) and not self.use_bf16
        self.scaler = GradScaler("cuda", enabled=self.use_fp16)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=config.get("label_smoothing", 0.0),
        )
        # Eval always uses raw cross-entropy (no smoothing) so val_loss is comparable
        # across trials that sweep label_smoothing as a hyperparameter.
        self.eval_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        os.makedirs(config["output_dir"], exist_ok=True)

        # Wandb setup
        self.use_wandb = self.logging_config.get("use_wandb", False)
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            # Support both WANDB_API_KEY and wandb_api_key env var names
            api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("wandb_api_key")
            if api_key:
                wandb.login(key=api_key, relogin=True)
            wandb.init(
                project=self.logging_config.get("project_name", "fol-slm"),
                name=self.logging_config.get("run_name"),
                config={**config, **(diagnostics_config or {})},
            )
        else:
            self.wandb = None

    def train(self, start_step: int = 0, trial_callback=None):
        """Main training loop.

        Args:
            start_step: Step to resume from (used after checkpoint resume).
            trial_callback: Optional callable(step, val_loss) -> bool.
                If it returns True, training stops early (for Optuna pruning).

        Returns:
            dict with final_val_loss and steps_completed.
        """
        cfg = self.config
        step = start_step
        self.optimizer.zero_grad()
        final_val_loss = 0.0
        final_val_entropy = 0.0

        while step < cfg["max_steps"]:
            # Curriculum: swap to the next DataLoader when a phase boundary is passed
            while self._curriculum_loaders and step >= self._curriculum_loaders[0][0]:
                until_step, next_loader = self._curriculum_loaders.pop(0)
                self.train_loader = next_loader
                print(f"[step {step}] Curriculum: switching to next phase loader (phase until step {until_step})")

            for batch in self.train_loader:
                if step >= cfg["max_steps"]:
                    break

                loss = self._train_step(batch, step)
                step += 1

                log_every = self.logging_config.get("log_every", cfg.get("log_every", 50))
                if step % log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"[step {step}] loss: {loss:.4f}  lr: {lr:.2e}")
                    if self.use_wandb:
                        log_dict = {"train/loss": loss, "train/lr": lr}
                        if self._last_cls_loss is not None:
                            log_dict["train/cls_loss"] = self._last_cls_loss
                        self.wandb.log(log_dict, step=step)

                    if self.diagnostics_config.get("enabled", False):
                        self._log_diagnostics(step)

                if step % cfg["eval_every"] == 0:
                    val_loss, val_entropy = self.evaluate()
                    final_val_loss = val_loss
                    final_val_entropy = val_entropy
                    print(f"[step {step}] val_loss: {val_loss:.4f}  val_entropy: {val_entropy:.4f}")
                    if self.use_wandb:
                        self.wandb.log({"val/loss": val_loss, "val/cross_attn_entropy": val_entropy}, step=step)

                    if trial_callback is not None:
                        should_prune = trial_callback(step, val_loss, val_entropy)
                        if should_prune:
                            if self.use_wandb:
                                self.wandb.finish()
                            return {"final_val_loss": val_loss, "final_val_entropy": val_entropy, "steps_completed": step, "pruned": True}

                if step % cfg["save_every"] == 0:
                    self._save(step)

        self._save("final")
        print("Training complete.")

        if self.use_wandb:
            self.wandb.finish()

        return {"final_val_loss": final_val_loss, "final_val_entropy": final_val_entropy, "steps_completed": step, "pruned": False}

    def _train_step(self, batch, step):
        cfg = self.config
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        decoder_input_ids = batch["decoder_input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        self._last_decoder_input_ids = decoder_input_ids

        # Build proof self-attention mask: at proof positions, block FOL premises from self-attention
        proof_mask = None
        if self.extra_id_2_id is not None and self.extra_id_3_id is not None:
            n_heads = self.model.decoder.layers[0].self_attn.num_heads
            proof_mask = build_proof_self_attn_mask(
                decoder_input_ids, self.extra_id_2_id, self.extra_id_3_id, n_heads
            )

        dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        with autocast("cuda", dtype=dtype, enabled=self.use_bf16 or self.use_fp16):
            logits = self.model(input_ids, attention_mask, decoder_input_ids, proof_mask)
            B, T, V = logits.shape
            # Compute loss in float32 for numerical stability
            loss = self.loss_fn(logits.float().view(B * T, V), labels.view(B * T))

            # Answer classification auxiliary loss at <extra_id_4> positions.
            # Always computed for logging; only added to loss after answer_cls_start_step.
            # coeff is auto-calibrated at answer_cls_start_step using EMA of CE and cls losses:
            #   coeff = target_ratio * ema_ce / ema_cls  (target_ratio set in config as answer_cls_coeff)
            cls_start = cfg.get("answer_cls_start_step", 0)
            if self.answer_cls_id is not None:
                cls_loss = self._compute_cls_loss(decoder_input_ids, labels)
                if cls_loss is not None:
                    self._last_cls_loss = cls_loss.item()
                    # Update EMAs before cls_start for calibration
                    ce_val = loss.item()
                    a = self._ema_alpha
                    self._ema_ce = ce_val if self._ema_ce is None else (1 - a) * self._ema_ce + a * ce_val
                    self._ema_cls = self._last_cls_loss if self._ema_cls is None else (1 - a) * self._ema_cls + a * self._last_cls_loss
                    # At cls_start: auto-calibrate coeff so cls contributes target_ratio of CE magnitude
                    if step == cls_start and cfg.get("answer_cls_coeff", 0.0) > 0.0:
                        target_ratio = cfg["answer_cls_coeff"]  # e.g. 0.05 = cls is 5% of CE
                        calibrated = target_ratio * self._ema_ce / self._ema_cls
                        cfg["_calibrated_cls_coeff"] = calibrated
                        print(f"  [step {step}] Auto-calibrated cls_coeff: {calibrated:.4f} "
                              f"(target_ratio={target_ratio}, ema_ce={self._ema_ce:.4f}, ema_cls={self._ema_cls:.4f})")
                        if self.use_wandb:
                            self.wandb.log({"train/calibrated_cls_coeff": calibrated}, step=step)
                    # Apply coeff: use calibrated value if available, else fall back to config value
                    active_coeff = cfg.get("_calibrated_cls_coeff", cfg.get("answer_cls_coeff", 0.0))
                    if active_coeff > 0.0 and step >= cls_start:
                        loss = loss + active_coeff * cls_loss
                else:
                    self._last_cls_loss = None

            loss = loss / cfg["grad_accum_steps"]

        if not torch.isfinite(loss):
            print(f"  [step {step}] WARNING: non-finite loss ({loss.item():.4f}), skipping batch")
            self.optimizer.zero_grad()
            return 0.0

        self.scaler.scale(loss).backward()

        if (step + 1) % cfg["grad_accum_steps"] == 0:
            if self.use_fp16:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg["clip_grad_norm"])
            if self.use_fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return loss.item() * cfg["grad_accum_steps"]

    def _compute_cls_loss(self, decoder_input_ids, labels):
        """Auxiliary 3-way classification loss at <extra_id_4> positions.

        At each position where decoder_input_ids == answer_cls_id, take the
        decoder hidden state and predict True(0)/False(1)/Unknown(2) directly.
        Scans labels from that position forward to find the answer token.
        """
        hidden = getattr(self.model.decoder, "last_hidden", None)
        if hidden is None:
            return None

        cls_hiddens, cls_targets = [], []
        B = decoder_input_ids.shape[0]

        for b in range(B):
            positions = (decoder_input_ids[b] == self.answer_cls_id).nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                continue
            pos = positions[0].item()
            # Scan labels from pos onward to find True/False/Unknown token
            cls_idx = None
            for offset in range(min(5, labels.shape[1] - pos)):
                tok = labels[b, pos + offset].item()
                if tok in self.answer_tok_to_cls:
                    cls_idx = self.answer_tok_to_cls[tok]
                    break
            if cls_idx is None:
                continue
            cls_hiddens.append(hidden[b, pos].float())
            cls_targets.append(cls_idx)

        if not cls_hiddens:
            return None

        h = torch.stack(cls_hiddens)                                        # (N, d_model)
        t = torch.tensor(cls_targets, device=h.device, dtype=torch.long)   # (N,)
        cls_logits = self.model.answer_cls_head(h)                          # (N, 3)
        return torch.nn.functional.cross_entropy(cls_logits, t)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        total_entropy = 0.0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            decoder_input_ids = batch["decoder_input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            proof_mask = None
            if self.extra_id_2_id is not None and self.extra_id_3_id is not None:
                n_heads = self.model.decoder.layers[0].self_attn.num_heads
                proof_mask = build_proof_self_attn_mask(
                    decoder_input_ids, self.extra_id_2_id, self.extra_id_3_id, n_heads
                )
            dtype = torch.bfloat16 if self.use_bf16 else torch.float16
            with autocast("cuda", dtype=dtype, enabled=self.use_bf16 or self.use_fp16):
                logits = self.model(input_ids, attention_mask, decoder_input_ids, proof_mask)
            B, T, V = logits.shape
            loss = self.eval_loss_fn(logits.float().view(B * T, V), labels.view(B * T))
            # Weight by token count so variable-length batches contribute proportionally
            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            # Entropy weighted by token count for consistency with val_loss
            weights = getattr(self.model.decoder, "cross_attn_weights", None)
            if weights is not None:
                fol_mask = None
                if self.proof_sentinel_id is not None:
                    fol_mask = build_fol_mask(decoder_input_ids, self.proof_sentinel_id)
                total_entropy += cross_attention_entropy(weights, fol_mask) * n_tokens

        val_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        val_entropy = total_entropy / total_tokens if total_tokens > 0 else 0.0
        return val_loss, val_entropy

    def _log_diagnostics(self, step):
        diag = self.diagnostics_config

        metrics = {}

        if diag.get("log_grad_norms", False):
            total_norm = 0.0
            for p in self.model.parameters():
                if p.requires_grad and p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            metrics["diagnostics/grad_norm"] = total_norm

        if diag.get("log_cross_attn_entropy", False):
            weights = getattr(self.model.decoder, "cross_attn_weights", None)
            if weights is not None:
                fol_mask = None
                if self.proof_sentinel_id is not None and self._last_decoder_input_ids is not None:
                    fol_mask = build_fol_mask(self._last_decoder_input_ids, self.proof_sentinel_id)
                entropy = cross_attention_entropy(weights, fol_mask)
                metrics["diagnostics/cross_attn_entropy"] = entropy

        if metrics:
            if self.use_wandb:
                self.wandb.log(metrics, step=step)
            for k, v in metrics.items():
                print(f"  [{step}] {k}: {v:.4f}")

    def _save(self, tag):
        path = os.path.join(self.config["output_dir"], f"checkpoint_{tag}.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "step": tag if isinstance(tag, int) else -1,
        }, path)
        print(f"Saved checkpoint: {path}")

    def resume_from(self, checkpoint_path: str, reset_scheduler: bool = False) -> int:
        """Load checkpoint and return the step to resume from.

        Args:
            checkpoint_path: Path to the checkpoint file.
            reset_scheduler: If True, rebuild the LR scheduler from the current config
                instead of restoring the saved scheduler state. Use this when extending
                max_steps after a plateau — the new schedule will decay more slowly,
                keeping LR higher for longer from the resumed position.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        step = checkpoint.get("step", 0)
        if step == -1:
            step = 0

        if reset_scheduler:
            # Rebuild scheduler from updated config and fast-forward to current position.
            # This is used when max_steps has been extended: the old saved scheduler state
            # is anchored to the original total_steps and would decay too fast.
            cfg = self.config
            grad_accum = cfg["grad_accum_steps"]
            optimizer_steps = cfg["max_steps"] // grad_accum
            warmup_optimizer_steps = cfg["warmup_steps"] // grad_accum
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_optimizer_steps,
                num_training_steps=optimizer_steps,
            )
            # Fast-forward to match completed optimizer steps
            completed_optimizer_steps = step // grad_accum
            for _ in range(completed_optimizer_steps):
                self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            print(f"Scheduler rebuilt: {optimizer_steps} total opt-steps, "
                  f"fast-forwarded to opt-step {completed_optimizer_steps}, LR={lr:.2e}")
        else:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        print(f"Resumed from {checkpoint_path} at step {step}")
        return step
