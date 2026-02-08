"""
Training Tracker for PDMBenchmark

Provides real-time training progress tracking with:
- Epoch-by-epoch metrics logging
- Training history storage
- Progress callbacks for UI updates
- Results persistence in JSON format
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    test_loss: float = 0.0
    test_accuracy: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TrainingRun:
    """Complete training run information."""
    run_id: str
    model: str
    dataset: str
    start_time: str = ""
    end_time: str = ""
    status: str = "pending"  # pending, running, completed, failed, stopped
    total_epochs: int = 0
    current_epoch: int = 0
    best_val_accuracy: float = 0.0
    best_epoch: int = 0
    config: Dict = field(default_factory=dict)
    epoch_history: List[Dict] = field(default_factory=list)
    final_metrics: Dict = field(default_factory=dict)
    error_message: str = ""

    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now().isoformat()
        if not self.run_id:
            self.run_id = f"{self.model}_{self.dataset}_{int(time.time())}"


class TrainingTracker:
    """
    Tracks training progress and provides callbacks for UI updates.

    Usage:
        tracker = TrainingTracker()
        tracker.start_run(model='TimesNet', dataset='01', config={...})

        for epoch in range(epochs):
            # ... training code ...
            tracker.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

        tracker.finish_run(final_metrics)

        # For UI updates
        tracker.set_progress_callback(lambda progress: update_ui(progress))
    """

    def __init__(self, results_dir: str = './results/training_history/'):
        """Initialize the training tracker."""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.current_run: Optional[TrainingRun] = None
        self._progress_callbacks: List[Callable] = []
        self._stop_requested = False

    def set_progress_callback(self, callback: Callable[[Dict], None]):
        """Set a callback function to receive progress updates."""
        self._progress_callbacks.append(callback)

    def clear_callbacks(self):
        """Clear all progress callbacks."""
        self._progress_callbacks.clear()

    def _notify_progress(self):
        """Notify all callbacks with current progress."""
        if self.current_run is None:
            return

        progress = {
            'run_id': self.current_run.run_id,
            'model': self.current_run.model,
            'dataset': self.current_run.dataset,
            'status': self.current_run.status,
            'current_epoch': self.current_run.current_epoch,
            'total_epochs': self.current_run.total_epochs,
            'progress_pct': (self.current_run.current_epoch / max(self.current_run.total_epochs, 1)) * 100,
            'best_val_accuracy': self.current_run.best_val_accuracy,
            'best_epoch': self.current_run.best_epoch,
            'epoch_history': self.current_run.epoch_history,
            'stop_requested': self._stop_requested,
        }

        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                print(f"Error in progress callback: {e}")

    def start_run(
        self,
        model: str,
        dataset: str,
        total_epochs: int,
        config: Dict = None
    ) -> str:
        """Start tracking a new training run."""
        self._stop_requested = False

        self.current_run = TrainingRun(
            run_id=f"{model}_{dataset}_{int(time.time())}",
            model=model,
            dataset=dataset,
            total_epochs=total_epochs,
            status="running",
            config=config or {}
        )

        self._notify_progress()
        return self.current_run.run_id

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        test_loss: float = 0.0,
        test_accuracy: float = 0.0,
        learning_rate: float = 0.0,
        epoch_time: float = 0.0
    ) -> bool:
        """
        Log metrics for a training epoch.

        Returns:
            bool: False if stop was requested, True otherwise
        """
        if self.current_run is None:
            raise RuntimeError("No training run started. Call start_run() first.")

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            learning_rate=learning_rate,
            epoch_time=epoch_time
        )

        self.current_run.epoch_history.append(asdict(metrics))
        self.current_run.current_epoch = epoch + 1

        # Track best validation accuracy
        if val_accuracy > self.current_run.best_val_accuracy:
            self.current_run.best_val_accuracy = val_accuracy
            self.current_run.best_epoch = epoch + 1

        self._notify_progress()

        return not self._stop_requested

    def request_stop(self):
        """Request training to stop."""
        self._stop_requested = True
        if self.current_run:
            self.current_run.status = "stopping"
        self._notify_progress()

    def is_stop_requested(self) -> bool:
        """Check if stop has been requested."""
        return self._stop_requested

    def finish_run(self, final_metrics: Dict = None, success: bool = True, error_message: str = ""):
        """Finish the current training run."""
        if self.current_run is None:
            return

        self.current_run.end_time = datetime.now().isoformat()

        if self._stop_requested:
            self.current_run.status = "stopped"
        elif success:
            self.current_run.status = "completed"
        else:
            self.current_run.status = "failed"
            self.current_run.error_message = error_message

        if final_metrics:
            self.current_run.final_metrics = final_metrics

        # Save to file
        self._save_run()
        self._notify_progress()

    def _save_run(self):
        """Save the current run to a JSON file."""
        if self.current_run is None:
            return

        run_file = os.path.join(
            self.results_dir,
            f"{self.current_run.run_id}.json"
        )

        with open(run_file, 'w') as f:
            json.dump(asdict(self.current_run), f, indent=2, default=str)

    def load_run(self, run_id: str) -> Optional[TrainingRun]:
        """Load a training run from file."""
        run_file = os.path.join(self.results_dir, f"{run_id}.json")
        if not os.path.exists(run_file):
            return None

        with open(run_file, 'r') as f:
            data = json.load(f)
            return TrainingRun(**data)

    def get_all_runs(self) -> List[Dict]:
        """Get all training runs."""
        runs = []
        if not os.path.exists(self.results_dir):
            return runs

        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.results_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        runs.append(json.load(f))
                except Exception:
                    continue

        # Sort by start time, newest first
        runs.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return runs

    def get_run_summary(self) -> List[Dict]:
        """Get summary of all runs for display."""
        runs = self.get_all_runs()
        summaries = []

        for run in runs:
            summaries.append({
                'run_id': run.get('run_id', 'unknown'),
                'model': run.get('model', 'unknown'),
                'dataset': run.get('dataset', 'unknown'),
                'status': run.get('status', 'unknown'),
                'best_val_accuracy': run.get('best_val_accuracy', 0),
                'total_epochs': run.get('total_epochs', 0),
                'start_time': run.get('start_time', ''),
                'end_time': run.get('end_time', ''),
            })

        return summaries

    def get_current_progress(self) -> Optional[Dict]:
        """Get current training progress."""
        if self.current_run is None:
            return None

        return {
            'run_id': self.current_run.run_id,
            'model': self.current_run.model,
            'dataset': self.current_run.dataset,
            'status': self.current_run.status,
            'current_epoch': self.current_run.current_epoch,
            'total_epochs': self.current_run.total_epochs,
            'progress_pct': (self.current_run.current_epoch / max(self.current_run.total_epochs, 1)) * 100,
            'best_val_accuracy': self.current_run.best_val_accuracy,
            'epoch_history': self.current_run.epoch_history,
        }


# Global tracker instance
_global_tracker = None


def get_tracker() -> TrainingTracker:
    """Get the global training tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TrainingTracker()
    return _global_tracker