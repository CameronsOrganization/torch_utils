import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange
from accelerate import Accelerator

from typing import Optional, Callable, Iterable

import experiment_utils as eu

from .log import log_params, log_values, log_value


def test_step():
    pass


def test(
    model: nn.Module,
    test_loader: Iterable,
    test_step: Callable = test_step,
    run_name: Optional[str] = None,
    logger: Optional[eu.Logger] = None,
):
    model.eval()
    pass


def train_step(
    batch: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict,
    accelerator: Accelerator,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    optimizer.zero_grad()
    output = model(batch)
    loss = output["loss"]
    accelerator.backward(loss)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    # Update metrics
    metrics["steps"] += 1
    metrics["loss"] += loss.item()
    for key in model.metrics.keys():
        metrics[key] += output[key].item()


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: Iterable,
    val_loader: Optional[Iterable] = None,
    train_step: Callable = train_step,
    val: Callable = test,
    logger=None,
    train_steps: int = 10000,
    log_steps: int = 100,
    val_steps: int = 1000,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    # Log params
    log_params(
        logger,
        train_steps=train_steps,
        log_steps=log_steps,
        val_steps=val_steps,
        lr=optimizer.defaults["lr"],
    )

    # Accelerator
    accelerator = Accelerator()
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)
    model.train()

    # Initialize metrics
    metrics = {
        **{"steps": 0, "loss": 0},
        **{key: 0 for key in model.metrics.keys()},
    }
    step = 0
    progress = trange(train_steps, desc="Training")

    while step < train_steps:
        for batch in train_loader:
            # Next step
            step += 1
            progress.update(1)
            log_values(step, logger, lr=optimizer.param_groups[0]["lr"])

            # Train step
            train_step(
                batch=batch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                accelerator=accelerator,
            )

            # Validation
            if val_loader is not None and step % val_steps == 0:
                val()
                model.train()
            # Log
            if step % log_steps == 0:
                # Log loss
                log_value(
                    "train_loss",
                    metrics["loss"] / metrics["steps"],
                    step,
                    logger,
                    eu.compare_fns.min,
                )
                # Log model specific metrics
                for key in model.metrics.keys():
                    value = metrics[key] / metrics["steps"]
                    log_value(
                        f"train_{key}",
                        value,
                        step,
                        logger,
                        compare_fn=model.metrics[key],
                    )
                # Reset metrics
                metrics = {
                    **{"steps": 0, "loss": 0},
                    **{key: 0 for key in model.metrics.keys()},
                }
            # End of training
            if train_steps <= step:
                break
    # Final validation
    if val_loader is not None and step % val_steps != 0:
        val()


# class Runner:
#     def __init__(self, model, optimizer, device, logger: eu.Logger = None):
#         self.model = model
#         self.optimizer = optimizer
#         self.device = device
#         self.logger = logger

#     def train(
#         self,
#         train_loader,
#         steps: int,
#         val_loader=None,
#         log_steps: int = None,
#         val_steps: int = None,
#     ):
#         val_steps = val_steps if val_steps is not None else max(steps // 100, 1)
#         if self.logger is not None:
#             log_steps = log_steps if log_steps is not None else max(steps // 100, 1)
#             self.logger.log_params(
#                 {
#                     "steps": steps,
#                     "model": self.model.__class__.__name__,
#                     "optimizer": self.optimizer.__class__.__name__,
#                     "device": self.device,
#                     "learning_rate": self.optimizer.param_groups[0]["lr"],
#                 }
#             )
#         step = 0
#         progress_bar = trange(steps, desc="Training", leave=False)
#         while step < steps:
#             for samples in train_loader:
#                 step += 1
#                 progress_bar.update(1)
#                 if self.logger is not None:
#                     self.logger.log_values(
#                         {
#                             "step": step,
#                             "learning_rate": self.optimizer.param_groups[0]["lr"],
#                         },
#                         step,
#                     )

#                 train_loss = self.train_step(samples)
#                 if self.logger is not None and step % log_steps == 0 and step > 0:
#                     self.logger.log_value(
#                         "train_loss",
#                         train_loss,
#                         step=step,
#                         compare_fn=eu.compare_fns.min,
#                     )

#                 if step % val_steps == 0 and step > 0 and val_loader is not None:
#                     val_loss = self.test(val_loader, "val", step)
#                 if step >= steps:
#                     break
#         if val_loader is not None and step % val_steps != 0:
#             val_loss = self.test(val_loader, "val", step)

#         return val_loss

#     def train_step(self, samples):
#         self.model.to(self.device)
#         self.model.train()
#         self.optimizer.zero_grad()
#         samples = {k: v.to(self.device) for k, v in samples.items()}
#         outputs = self.model(samples)
#         loss = outputs["loss"]
#         loss.backward()
#         self.optimizer.step()
#         return loss.item()

#     @torch.no_grad()
#     def test(
#         self, data_loader, split: Optional[str] = "test", step: Optional[int] = None
#     ):
#         self.model.to(self.device)
#         self.model.eval()
#         total_loss = 0
#         total_accuracy = 0
#         total_samples = 0

#         for samples in data_loader:
#             bs = samples[next(iter(samples.keys()))].size(0)
#             outputs = self.test_step(samples)
#             total_loss += outputs["loss"].item() * bs
#             total_accuracy += outputs["accuracy"].item() * bs
#             total_samples += bs

#         avg_loss = total_loss / total_samples
#         avg_accuracy = total_accuracy / total_samples
#         if self.logger is not None:
#             self.logger.log_value(
#                 f"{split}_loss", avg_loss, step=step, compare_fn=eu.compare_fns.min
#             )
#             self.logger.log_value(
#                 f"{split}_accuracy",
#                 avg_accuracy,
#                 step=step,
#                 compare_fn=eu.compare_fns.max,
#             )
#         return avg_loss

#     @torch.no_grad()
#     def test_step(self, samples) -> float:
#         self.model.eval()
#         samples = {k: v.to(self.device) for k, v in samples.items()}
#         outputs = self.model(samples)
#         return outputs
