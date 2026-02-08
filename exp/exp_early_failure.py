"""
Early Failure Prediction Experiment Class.

Supports multiple prediction modes:
- classification: Binary failure/no-failure prediction
- rul: Remaining Useful Life regression
- hazard: Multi-horizon hazard prediction
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy, cal_f1, evaluate_calibration
from utils.early_failure_metrics import (
    evaluate_rul_predictions,
    evaluate_failure_predictions,
    evaluate_survival_predictions,
    nasa_scoring_function,
    concordance_index
)
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import tqdm
from typing import Dict, Tuple, Optional

warnings.filterwarnings('ignore')


class HuberLoss(nn.Module):
    """Huber loss for RUL prediction (smooth L1)."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(y_pred - y_true)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss for RUL prediction.
    Penalizes late predictions more than early predictions.
    """

    def __init__(self, alpha: float = 1.5):
        """
        Args:
            alpha: Penalty factor for late predictions (>1 means more penalty for late)
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_pred - y_true  # positive = late prediction

        loss = torch.where(
            diff >= 0,
            self.alpha * diff ** 2,  # Late prediction (overestimate RUL)
            diff ** 2                 # Early prediction (underestimate RUL)
        )
        return loss.mean()


class HazardHead(nn.Module):
    """Multi-horizon hazard prediction head."""

    def __init__(self, input_dim: int, horizons: list):
        super().__init__()
        self.horizons = horizons
        self.n_horizons = len(horizons)

        # Separate output for each horizon
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, 1)
            for _ in horizons
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Hazard probabilities (batch_size, n_horizons)
        """
        outputs = [torch.sigmoid(head(x)) for head in self.heads]
        return torch.cat(outputs, dim=-1)


class Exp_EarlyFailure(Exp_Basic):
    """
    Experiment class for early failure prediction.

    Supports three prediction modes:
    - classification: Binary failure prediction
    - rul: Remaining Useful Life regression
    - hazard: Multi-horizon hazard probabilities
    """

    def __init__(self, args):
        self.failure_mode = getattr(args, 'failure_prediction_mode', 'classification') or 'classification'
        self.prediction_horizons = getattr(args, 'prediction_horizons', [10, 20, 50, 100])
        self.rul_threshold = getattr(args, 'rul_threshold', None)
        super(Exp_EarlyFailure, self).__init__(args)

    def _build_model(self):
        """Build model with appropriate head for failure prediction."""
        print('Loading dataset for early failure prediction!')

        # Load data
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        test_data, test_loader = self._get_data(flag='TEST')

        self.train_data = train_data
        self.train_loader = train_loader
        self.vali_data = vali_data
        self.vali_loader = vali_loader
        self.test_data = test_data
        self.test_loader = test_loader

        # Check if dataset has true RUL values
        self.has_rul = getattr(train_data, 'has_rul', False)
        print(f"Dataset has true RUL values: {self.has_rul}")

        # Validate that RUL/hazard modes are only used with datasets that have RUL
        if self.failure_mode in ['rul', 'hazard'] and not self.has_rul:
            raise ValueError(
                f"Failure mode '{self.failure_mode}' requires a dataset with true RUL values. "
                f"This dataset does not have RUL data. Use 'classification' mode instead."
            )

        # Set model parameters
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = self.args.seq_len  # Set to seq_len for model compatibility
        self.args.enc_in = train_data.feature_df.shape[1]

        # Adjust num_class based on failure mode
        if self.failure_mode == 'classification':
            self.args.num_class = 2  # Binary: fail/no-fail
        elif self.failure_mode == 'rul':
            self.args.num_class = 1  # Regression output
        elif self.failure_mode == 'hazard':
            self.args.num_class = len(self.prediction_horizons)

        # Original task name for model compatibility
        original_task = self.args.task_name
        self.args.task_name = 'classification'  # Use classification architecture

        # Build base model
        model = self.model_dict[self.args.model].Model(self.args).float()

        # Restore task name
        self.args.task_name = original_task

        # Initialize LazyLinear layers with a dummy forward pass
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(self.train_loader))[0][:2].float()
            try:
                _ = model(sample_batch, None, None, None)
            except Exception:
                pass  # Some models might fail on initialization, but LazyLinear should be initialized
        model.train()

        # Add custom head if needed (for RUL/hazard modes)
        if self.failure_mode == 'rul':
            # Replace final layer for regression
            self._modify_output_layer(model, output_dim=1)
        elif self.failure_mode == 'hazard':
            # Add hazard prediction head
            self._modify_output_layer(model, output_dim=len(self.prediction_horizons))

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params}")
        print(f"Failure prediction mode: {self.failure_mode}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _modify_output_layer(self, model, output_dim: int):
        """Modify the model's output layer for different prediction modes."""
        # Find and replace the projection/final layer
        if hasattr(model, 'projection'):
            in_features = model.projection.in_features
            model.projection = nn.Linear(in_features, output_dim)
        elif hasattr(model, 'head'):
            if hasattr(model.head, 'linear'):
                in_features = model.head.linear.in_features
                model.head.linear = nn.Linear(in_features, output_dim)

    def _get_data(self, flag):
        """Load data for early failure prediction."""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """Select optimizer."""
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """Select loss function based on failure mode."""
        if self.failure_mode == 'classification':
            return nn.CrossEntropyLoss()
        elif self.failure_mode == 'rul':
            # Use Huber loss for RUL (more robust to outliers)
            return HuberLoss(delta=10.0)
        elif self.failure_mode == 'hazard':
            # Use BCEWithLogitsLoss - more stable, applies sigmoid internally
            return nn.BCEWithLogitsLoss()
        else:
            return nn.MSELoss()

    def train(self):
        """Train the early failure prediction model."""
        train_data = self.train_data
        train_loader = self.train_loader
        vali_data = self.vali_data
        vali_loader = self.vali_loader
        test_data = self.test_data
        test_loader = self.test_loader

        path = os.path.join(
            self.args.checkpoints,
            self.args.root_path.split("/")[-2],
            f"{self.args.model}_early_failure_{self.failure_mode}"
        )
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        epoch_time_list = []

        for epoch in tqdm.tqdm(range(self.args.train_epochs)):
            train_loss = []
            preds = []
            trues = []

            self.model.train()
            start_time = time.time()

            for i, batch_data in enumerate(train_loader):
                model_optim.zero_grad()

                # Handle both 2-value and 3-value returns from data loader
                if len(batch_data) == 3:
                    batch_x, label, rul = batch_data
                    rul = rul.to(self.device)
                else:
                    batch_x, label = batch_data
                    rul = None

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                # Forward pass
                outputs = self.model(batch_x, None, None, None)

                # Compute loss based on failure mode
                if self.failure_mode == 'classification':
                    # Convert to binary labels if needed
                    binary_labels = self._to_binary_labels(label, rul)
                    loss = criterion(outputs, binary_labels.long().squeeze(-1))
                elif self.failure_mode == 'rul':
                    # RUL regression - use true RUL values
                    rul_labels = self._to_rul_labels(label, rul)
                    loss = criterion(outputs.squeeze(), rul_labels.float())
                elif self.failure_mode == 'hazard':
                    # Multi-horizon hazard - use true RUL values
                    hazard_labels = self._to_hazard_labels(label, rul).to(outputs.device).float()
                    loss = criterion(outputs, hazard_labels)
                else:
                    loss = criterion(outputs, label.float())

                train_loss.append(loss.item())
                preds.append(outputs.detach().cpu())
                # Store RUL if available, otherwise store label
                if rul is not None and self.failure_mode in ['rul', 'hazard']:
                    trues.append(rul.detach().cpu())
                else:
                    trues.append(label.detach().cpu())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            epoch_time_list.append(time.time() - start_time)

            # Calculate training metrics
            train_metrics = self._calculate_metrics(preds, trues)
            train_loss = np.average(train_loss)

            # Validation
            val_loss, val_metrics = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Metric: {val_metrics['primary']:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Metric: {test_metrics['primary']:.4f}")

            # Log to wandb
            self.wandb.log({
                "Epoch": epoch + 1,
                "Loss/Train": train_loss,
                "Loss/Val": val_loss,
                "Loss/Test": test_loss,
                f"Metric/Train_{self._get_primary_metric_name()}": train_metrics['primary'],
                f"Metric/Val_{self._get_primary_metric_name()}": val_metrics['primary'],
                f"Metric/Test_{self._get_primary_metric_name()}": test_metrics['primary'],
                "Time Per Epoch": epoch_time_list[-1]
            }, commit=True)

            # Early stopping based on validation metric
            early_stopping(val_metrics['primary'], self.model, path)
            if early_stopping.early_stop:
                print(f"Current epoch: {epoch + 1}")
                print("Early stopping")
                break

        # Load best model
        best_model_path = path + '/checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        print(f"Average training time per epoch: {np.mean(epoch_time_list):.2f}s")
        return self.model

    def vali(self, vali_data, vali_loader, criterion) -> Tuple[float, Dict]:
        """Validate the model."""
        total_loss = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                # Handle both 2-value and 3-value returns from data loader
                if len(batch_data) == 3:
                    batch_x, label, rul = batch_data
                    rul = rul.to(self.device)
                else:
                    batch_x, label = batch_data
                    rul = None

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, None, None, None)

                # Compute loss
                if self.failure_mode == 'classification':
                    binary_labels = self._to_binary_labels(label, rul)
                    loss = criterion(outputs, binary_labels.long().squeeze(-1))
                elif self.failure_mode == 'rul':
                    rul_labels = self._to_rul_labels(label, rul)
                    loss = criterion(outputs.squeeze(), rul_labels.float())
                elif self.failure_mode == 'hazard':
                    # Ensure both tensors are on same device
                    hazard_labels = self._to_hazard_labels(label, rul).to(outputs.device).float()
                    loss = criterion(outputs, hazard_labels)
                else:
                    loss = criterion(outputs, label.float())

                total_loss.append(loss.item())
                preds.append(outputs.detach().cpu())
                # Store RUL if available for RUL/hazard modes
                if rul is not None and self.failure_mode in ['rul', 'hazard']:
                    trues.append(rul.detach().cpu())
                else:
                    trues.append(label.detach().cpu())

        total_loss = np.average(total_loss)
        metrics = self._calculate_metrics(preds, trues)

        return total_loss, metrics

    def test(self, load_model: bool = False):
        """Test the model and compute comprehensive metrics."""
        if load_model:
            print('Loading model')
            path = os.path.join(
                self.args.checkpoints,
                self.args.root_path.split("/")[-2],
                f"{self.args.model}_early_failure_{self.failure_mode}",
                'checkpoint.pth'
            )
            self.model.load_state_dict(torch.load(path, map_location=self.device))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, batch_data in tqdm.tqdm(enumerate(self.test_loader)):
                # Handle both 2-value and 3-value returns from data loader
                if len(batch_data) == 3:
                    batch_x, label, rul = batch_data
                    rul = rul.to(self.device)
                else:
                    batch_x, label = batch_data
                    rul = None

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, None, None, None)

                preds.append(outputs.detach().cpu())
                # Store RUL if available for RUL/hazard modes
                if rul is not None and self.failure_mode in ['rul', 'hazard']:
                    trues.append(rul.detach().cpu())
                else:
                    trues.append(label.detach().cpu())

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print(f'Test shape: {preds.shape}, {trues.shape}')

        # Calculate comprehensive metrics based on failure mode
        if self.failure_mode == 'classification':
            metrics = self._evaluate_classification(preds, trues)
        elif self.failure_mode == 'rul':
            metrics = self._evaluate_rul(preds, trues)
        elif self.failure_mode == 'hazard':
            metrics = self._evaluate_hazard(preds, trues)
        else:
            metrics = {}

        # Save results
        folder_path = f'./results/{self.args.root_path.split("/")[-2]}/{self.args.model}_early_failure_{self.failure_mode}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print("\n" + "=" * 50)
        print(f"EARLY FAILURE PREDICTION RESULTS ({self.failure_mode})")
        print("=" * 50)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")

        # Save to file
        file_name = 'result_early_failure.txt'
        args_dict = vars(self.args)
        with open(os.path.join(folder_path, file_name), 'w') as f:
            f.write(f"Failure Prediction Mode: {self.failure_mode}\n")
            f.write("=" * 50 + "\n\n")

            f.write("Arguments:\n")
            for key, value in args_dict.items():
                f.write(f"  {key}: {value}\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("Results:\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")

        return metrics

    def _to_binary_labels(self, labels: torch.Tensor, rul: torch.Tensor = None) -> torch.Tensor:
        """Convert labels to binary (fail/no-fail).

        Args:
            labels: Class labels (0 = healthy, >0 = failure)
            rul: Optional true RUL values (0-1 for percentage, or cycles)
        """
        if rul is not None and self.rul_threshold is not None:
            # Use true RUL values with threshold
            return (rul < self.rul_threshold).long()
        # Use class labels: class > 0 means failure
        return (labels > 0).long()

    def _to_rul_labels(self, labels: torch.Tensor, rul: torch.Tensor = None) -> torch.Tensor:
        """Convert to RUL values for regression.

        Args:
            labels: Class labels (not used if rul is provided)
            rul: True RUL values (required for RUL mode)
        """
        if rul is not None:
            # Use true RUL values
            return rul.float().squeeze()
        # Fallback to labels (should not happen if has_rul check is working)
        return labels.float().squeeze()

    def _to_hazard_labels(self, labels: torch.Tensor, rul: torch.Tensor = None) -> torch.Tensor:
        """Convert to multi-horizon hazard labels.

        Args:
            labels: Class labels (not used if rul is provided)
            rul: True RUL values (required for hazard mode)
        """
        if rul is not None:
            rul_values = rul.float().squeeze()
        else:
            # Fallback (should not happen if has_rul check is working)
            rul_values = labels.float().squeeze()

        batch_size = rul_values.shape[0] if rul_values.dim() > 0 else 1
        if rul_values.dim() == 0:
            rul_values = rul_values.unsqueeze(0)

        hazard_labels = torch.zeros(batch_size, len(self.prediction_horizons))
        for i, horizon in enumerate(self.prediction_horizons):
            # For percentage RUL (0-1), convert horizon to percentage
            # horizon of 0.1 means 10% of life remaining
            hazard_labels[:, i] = (rul_values <= horizon).float()

        return hazard_labels

    def _calculate_metrics(self, preds: list, trues: list) -> Dict:
        """Calculate metrics during training/validation.

        Note: For RUL/hazard modes, trues should already contain RUL values
        (set during training/validation loop).
        """
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        if self.failure_mode == 'classification':
            probs = torch.nn.functional.softmax(preds, dim=1)
            predictions = torch.argmax(probs, dim=1).numpy()
            # For classification, trues are class labels
            binary_trues = self._to_binary_labels(trues, None).numpy().flatten()
            accuracy = cal_accuracy(predictions, binary_trues)
            return {'primary': accuracy, 'accuracy': accuracy}

        elif self.failure_mode == 'rul':
            preds_np = preds.squeeze().numpy()
            # trues should already be RUL values (set in training loop)
            trues_np = trues.numpy().squeeze()
            rmse = np.sqrt(np.mean((preds_np - trues_np) ** 2))
            # Return negative RMSE as primary metric (higher is better for early stopping)
            return {'primary': -rmse, 'rmse': rmse}

        elif self.failure_mode == 'hazard':
            # Apply sigmoid to convert logits to probabilities (BCEWithLogitsLoss outputs logits)
            preds_np = torch.sigmoid(preds).numpy()
            # trues should already be RUL values (set in training loop)
            hazard_trues = self._to_hazard_labels(trues, trues).numpy()
            # Average AUC across horizons
            from sklearn.metrics import roc_auc_score
            aucs = []
            for i in range(preds_np.shape[1]):
                if len(np.unique(hazard_trues[:, i])) > 1:
                    aucs.append(roc_auc_score(hazard_trues[:, i], preds_np[:, i]))
            avg_auc = np.mean(aucs) if aucs else 0.0
            return {'primary': avg_auc, 'avg_auc': avg_auc}

        return {'primary': 0.0}

    def _get_primary_metric_name(self) -> str:
        """Get name of primary metric for logging."""
        if self.failure_mode == 'classification':
            return 'accuracy'
        elif self.failure_mode == 'rul':
            return 'neg_rmse'
        elif self.failure_mode == 'hazard':
            return 'avg_auc'
        return 'metric'

    def _evaluate_classification(self, preds: torch.Tensor, trues: torch.Tensor) -> Dict:
        """Comprehensive evaluation for classification mode."""
        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).numpy()
        # trues are class labels for classification mode
        binary_trues = self._to_binary_labels(trues, None).numpy().flatten()
        prob_positive = probs[:, 1].numpy() if probs.shape[1] > 1 else probs[:, 0].numpy()

        # Basic classification metrics
        accuracy = cal_accuracy(predictions, binary_trues)
        f1_micro, f1_macro, f1_weighted = cal_f1(predictions, binary_trues)
        nll, ece, brier = evaluate_calibration(probs.float(), torch.tensor(binary_trues).float())

        # Failure prediction specific metrics
        failure_metrics = evaluate_failure_predictions(binary_trues, prob_positive)

        metrics = {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'ece': ece,
            'nll': nll,
            'brier': brier,
            **failure_metrics
        }

        return metrics

    def _evaluate_rul(self, preds: torch.Tensor, trues: torch.Tensor) -> Dict:
        """Comprehensive evaluation for RUL mode.

        Note: trues should already be RUL values (set in test loop).
        """
        preds_np = preds.squeeze().numpy()
        # trues are already RUL values
        trues_np = trues.numpy().squeeze()

        # Get all RUL metrics
        rul_metrics = evaluate_rul_predictions(trues_np, preds_np)

        return rul_metrics

    def _evaluate_hazard(self, preds: torch.Tensor, trues: torch.Tensor) -> Dict:
        """Comprehensive evaluation for hazard mode.

        Note: trues should already be RUL values (set in test loop).
        """
        # Apply sigmoid to convert logits to probabilities (BCEWithLogitsLoss outputs logits)
        preds_np = torch.sigmoid(preds).numpy()
        # trues are already RUL values
        rul_values = trues.numpy().squeeze()
        hazard_trues = self._to_hazard_labels(trues, trues).numpy()

        metrics = {}

        # Per-horizon metrics
        from sklearn.metrics import roc_auc_score, average_precision_score
        for i, horizon in enumerate(self.prediction_horizons):
            prefix = f'horizon_{horizon}'
            y_true = hazard_trues[:, i]
            y_prob = preds_np[:, i]

            if len(np.unique(y_true)) > 1:
                metrics[f'{prefix}_auc'] = float(roc_auc_score(y_true, y_prob))
                metrics[f'{prefix}_ap'] = float(average_precision_score(y_true, y_prob))

            # Failure prediction metrics at this horizon
            horizon_metrics = evaluate_failure_predictions(y_true, y_prob)
            for k, v in horizon_metrics.items():
                metrics[f'{prefix}_{k}'] = v

        # Average metrics
        horizon_aucs = [v for k, v in metrics.items() if k.endswith('_auc')]
        if horizon_aucs:
            metrics['avg_auc'] = float(np.mean(horizon_aucs))

        # C-index using predicted risk at first horizon
        metrics['c_index'] = concordance_index(rul_values, preds_np[:, 0])

        return metrics
