import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange

from typing import Optional

import experiment_utils as eu


class Runner:
    def __init__(self, model, optimizer, device, logger: eu.Logger = None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger

    def train(
        self,
        train_loader,
        steps: int,
        val_loader=None,
        log_steps: int = None,
        val_steps: int = None,
    ):
        val_steps = val_steps if val_steps is not None else max(steps // 100, 1)
        if self.logger is not None:
            log_steps = log_steps if log_steps is not None else max(steps // 100, 1)
            self.logger.log_params(
                {
                    "steps": steps,
                    "model": self.model.__class__.__name__,
                    "optimizer": self.optimizer.__class__.__name__,
                    "device": self.device,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )
        step = 0
        progress_bar = trange(steps, desc="Training", leave=False)
        while step < steps:
            for samples in train_loader:
                step += 1
                progress_bar.update(1)
                if self.logger is not None:
                    self.logger.log_values(
                        {
                            "step": step,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                        },
                        step,
                    )

                train_loss = self.train_step(samples)
                if self.logger is not None and step % log_steps == 0 and step > 0:
                    self.logger.log_value(
                        "train_loss",
                        train_loss,
                        step=step,
                        compare_fn=eu.compare_fns.min,
                    )

                if step % val_steps == 0 and step > 0 and val_loader is not None:
                    val_loss = self.test(val_loader, "val", step)
                if step >= steps:
                    break
        if val_loader is not None and step % val_steps != 0:
            val_loss = self.test(val_loader, "val", step)

        return val_loss

    def train_step(self, samples):
        self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        samples = {k: v.to(self.device) for k, v in samples.items()}
        outputs = self.model(samples)
        loss = outputs["loss"]
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(
        self, data_loader, split: Optional[str] = "test", step: Optional[int] = None
    ):
        self.model.to(self.device)
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_samples = 0

        for samples in data_loader:
            bs = samples[next(iter(samples.keys()))].size(0)
            outputs = self.test_step(samples)
            total_loss += outputs["loss"].item() * bs
            total_accuracy += outputs["accuracy"].item() * bs
            total_samples += bs

        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        if self.logger is not None:
            self.logger.log_value(
                f"{split}_loss", avg_loss, step=step, compare_fn=eu.compare_fns.min
            )
            self.logger.log_value(
                f"{split}_accuracy",
                avg_accuracy,
                step=step,
                compare_fn=eu.compare_fns.max,
            )
        return avg_loss

    @torch.no_grad()
    def test_step(self, samples) -> float:
        self.model.eval()
        samples = {k: v.to(self.device) for k, v in samples.items()}
        outputs = self.model(samples)
        return outputs
