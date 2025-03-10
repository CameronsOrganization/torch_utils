import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange

from typing import Optional


class Runner:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, train_loader, steps: int, val_loader=None):

        step = 0
        while step < steps:
            for data, targets in train_loader:
                step += 1
                self.train_step(data, targets)
                if step >= steps:
                    break
            if val_loader is not None:
                self.test(val_loader, "val")

    def train_step(self, data, targets):
        self.model.train()
        self.optimizer.zero_grad()
        data, targets = data.to(self.device), targets.to(self.device)
        outputs = self.model(data)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def test(self, data_loader, split: Optional[str] = "test"):
        self.model.eval()
        total_loss = 0
        total_samples = 0

        for data, targets in data_loader:
            bs = data.size(0)
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.model(data)
            loss = self.loss_fn(outputs, targets)
            total_loss += loss.item() * bs
            total_samples += bs

        avg_loss = total_loss / total_samples

    @torch.no_grad()
    def test_step(self, inputs, targets) -> float:
        self.model.eval()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss.item()
