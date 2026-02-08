"""
PDM Trainers Module

Flexible training pipelines for PDM models.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass, field


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""
    # Optimizer settings
    optimizer: str = 'adam'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD

    # Scheduler settings
    scheduler: str = 'cosine'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    warmup_epochs: int = 0

    # Training settings
    epochs: int = 50
    patience: int = 10
    min_delta: float = 1e-4

    # Mixed precision
    use_amp: bool = False

    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_best_only: bool = True

    # Logging
    log_interval: int = 10
    verbose: bool = True


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta


class Trainer:
    """
    Flexible trainer for PDM models.

    Example:
        trainer = Trainer(model, train_loader, val_loader)
        history = trainer.fit(epochs=50)

        # With custom config
        config = TrainerConfig(
            optimizer='adamw',
            learning_rate=1e-4,
            scheduler='cosine',
            epochs=100
        )
        trainer = Trainer(model, train_loader, val_loader, config=config)
        history = trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainerConfig()

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Loss function
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.use_amp else None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )

        # History
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        # Best model state
        self.best_model_state = None
        self.best_val_loss = float('inf')

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        optimizers = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop,
        }

        opt_class = optimizers.get(self.config.optimizer.lower(), torch.optim.Adam)

        params = {
            'params': self.model.parameters(),
            'lr': self.config.learning_rate,
            'weight_decay': self.config.weight_decay
        }

        if self.config.optimizer.lower() == 'sgd':
            params['momentum'] = self.config.momentum

        return opt_class(**params)

    def _create_scheduler(self):
        """Create learning rate scheduler based on config."""
        schedulers = {
            'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs
            ),
            'step': lambda: torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            ),
            'plateau': lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience
            ),
            'exponential': lambda: torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            ),
            'none': lambda: None
        }

        scheduler_fn = schedulers.get(self.config.scheduler.lower(), schedulers['cosine'])
        return scheduler_fn()

    def fit(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            epochs: Number of epochs (overrides config if provided)

        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.epochs

        if self.config.verbose:
            print(f"Training on {self.device}")
            print(f"Optimizer: {self.config.optimizer}, LR: {self.config.learning_rate}")
            print(f"Scheduler: {self.config.scheduler}")
            print("-" * 50)

        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation phase
            if self.val_loader is not None:
                val_loss, val_acc = self._validate()
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()

                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Early stopping
                if self.early_stopping(val_loss):
                    if self.config.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

                if self.config.verbose:
                    lr = self.optimizer.param_groups[0]['lr']
                    print(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                        f"LR: {lr:.6f}"
                    )
            else:
                if self.config.verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
                    )

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)

            outputs = self.model(x)
            loss = self.criterion(outputs, y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, path: str):
        """Save a training checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
