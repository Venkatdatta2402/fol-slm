import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup


class Trainer:
    def __init__(self, model, train_loader, val_loader, config: dict, device: torch.device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["learning_rate"],
        )
        total_steps = config["max_steps"]
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=total_steps,
        )
        self.scaler = GradScaler(enabled=config["fp16"])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        os.makedirs(config["output_dir"], exist_ok=True)

    def train(self):
        cfg = self.config
        step = 0
        self.optimizer.zero_grad()

        while step < cfg["max_steps"]:
            for batch in self.train_loader:
                if step >= cfg["max_steps"]:
                    break

                loss = self._train_step(batch, step)
                step += 1

                if step % cfg.get("log_every", 50) == 0:
                    print(f"[step {step}] loss: {loss:.4f}")

                if step % cfg["eval_every"] == 0:
                    val_loss = self.evaluate()
                    print(f"[step {step}] val_loss: {val_loss:.4f}")

                if step % cfg["save_every"] == 0:
                    self._save(step)

        self._save("final")
        print("Training complete.")

    def _train_step(self, batch, step):
        cfg = self.config
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        decoder_input_ids = batch["decoder_input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        with autocast(enabled=cfg["fp16"]):
            logits = self.model(input_ids, attention_mask, decoder_input_ids)
            B, T, V = logits.shape
            loss = self.loss_fn(logits.view(B * T, V), labels.view(B * T))
            loss = loss / cfg["grad_accum_steps"]

        self.scaler.scale(loss).backward()

        if (step + 1) % cfg["grad_accum_steps"] == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg["clip_grad_norm"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return loss.item() * cfg["grad_accum_steps"]

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, n = 0.0, 0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            decoder_input_ids = batch["decoder_input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids, attention_mask, decoder_input_ids)
            B, T, V = logits.shape
            loss = self.loss_fn(logits.view(B * T, V), labels.view(B * T))
            total_loss += loss.item()
            n += 1
        return total_loss / n if n > 0 else 0.0

    def _save(self, tag):
        path = os.path.join(self.config["output_dir"], f"checkpoint_{tag}.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")
