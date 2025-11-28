"""
Training logger utility for structured training run management.

This module provides utilities for creating timestamped training run folders
and logging comprehensive training history in JSON format.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np


class TrainingLogger:
    """Manages structured logging of training runs with timestamped folders."""
    
    def __init__(
        self,
        base_dir: str = "models/training_runs",
        stage: str = "end",
        board_size: int = 10
    ):
        """
        Initialize the training logger.
        
        Args:
            base_dir: Base directory for training runs
            stage: Training stage (end, -2, -5)
            board_size: Board size
        """
        self.base_dir = base_dir
        self.stage = stage
        self.board_size = board_size
        
        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_folder = f"{timestamp}_{stage}_{board_size}x{board_size}"
        self.run_path = os.path.join(base_dir, self.run_folder)
        
        # Create folder
        os.makedirs(self.run_path, exist_ok=True)
        
        # Training history
        self.epochs_data: List[Dict] = []
        self.config_data: Dict[str, Any] = {}
        self.best_epoch: Optional[int] = None
        self.best_test_accuracy: float = 0.0
        self.total_training_time: float = 0.0
        
    def log_epoch(
        self,
        epoch: int,
        train_accuracy: float,
        test_accuracy: Optional[float],
        epoch_time: float
    ):
        """
        Log metrics for one epoch.
        
        Args:
            epoch: Epoch number (1-indexed)
            train_accuracy: Training accuracy percentage
            test_accuracy: Test accuracy percentage (None if not tested this epoch)
            epoch_time: Time taken for this epoch in seconds
        """
        epoch_data = {
            "epoch": epoch,
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy) if test_accuracy is not None else None,
            "epoch_time_seconds": float(epoch_time)
        }
        self.epochs_data.append(epoch_data)
        
        # Track best epoch
        if test_accuracy is not None and test_accuracy > self.best_test_accuracy:
            self.best_test_accuracy = test_accuracy
            self.best_epoch = epoch
            
    def set_config(
        self,
        model_hyperparameters: Dict[str, Any],
        training_config: Dict[str, Any],
        cuda_config: Dict[str, Any],
        dataset_info: Dict[str, Any]
    ):
        """
        Set the configuration data.
        
        Args:
            model_hyperparameters: GTM model hyperparameters
            training_config: Training configuration
            cuda_config: CUDA grid/block configuration
            dataset_info: Dataset information
        """
        self.config_data = {
            "model_hyperparameters": model_hyperparameters,
            "training_config": training_config,
            "cuda_config": cuda_config,
            "dataset_info": dataset_info,
            "timestamp": datetime.now().isoformat()
        }
        
    def save_training_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.run_path, "training_history.json")
        
        history_data = {
            "epochs": self.epochs_data,
            "best_epoch": self.best_epoch,
            "best_test_accuracy": self.best_test_accuracy,
            "total_training_time_seconds": self.total_training_time
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
            
        print(f"Training history saved to {history_path}")
        
    def save_config(self):
        """Save configuration to JSON file."""
        config_path = os.path.join(self.run_path, "config.json")
        
        with open(config_path, 'w') as f:
            json.dump(self.config_data, f, indent=2)
            
        print(f"Configuration saved to {config_path}")
        
    def save_summary(self, final_train_acc: float, final_test_acc: float):
        """
        Save human-readable summary to text file.
        
        Args:
            final_train_acc: Final training accuracy
            final_test_acc: Final test accuracy
        """
        summary_path = os.path.join(self.run_path, "summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TRAINING RUN SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Run Folder: {self.run_folder}\n")
            f.write(f"Timestamp: {self.config_data.get('timestamp', 'N/A')}\n\n")
            
            f.write("Configuration:\n")
            f.write("-" * 60 + "\n")
            model_params = self.config_data.get('model_hyperparameters', {})
            f.write(f"  Board Size: {self.board_size}x{self.board_size}\n")
            f.write(f"  Stage: {self.stage}\n")
            f.write(f"  Number of Clauses: {model_params.get('number_of_clauses', 'N/A')}\n")
            f.write(f"  Depth: {model_params.get('depth', 'N/A')}\n")
            f.write(f"  T: {model_params.get('T', 'N/A')}\n")
            f.write(f"  s: {model_params.get('s', 'N/A')}\n")
            f.write(f"  Message Size: {model_params.get('message_size', 'N/A')}\n\n")
            
            f.write("Training Results:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Total Epochs: {len(self.epochs_data)}\n")
            f.write(f"  Total Training Time: {self.total_training_time:.2f} seconds\n")
            f.write(f"  Average Time per Epoch: {self.total_training_time / max(len(self.epochs_data), 1):.2f} seconds\n\n")
            
            f.write(f"  Final Training Accuracy: {final_train_acc:.2f}%\n")
            f.write(f"  Final Test Accuracy: {final_test_acc:.2f}%\n\n")
            
            if self.best_epoch is not None:
                f.write(f"  Best Epoch: {self.best_epoch}\n")
                f.write(f"  Best Test Accuracy: {self.best_test_accuracy:.2f}%\n\n")
            
            # Epoch-by-epoch results (last 10 if too many)
            f.write("Epoch-by-Epoch Results:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Epoch':<8} {'Train Acc':<12} {'Test Acc':<12} {'Time (s)':<10}\n")
            f.write("-" * 60 + "\n")
            
            epochs_to_show = self.epochs_data[-10:] if len(self.epochs_data) > 10 else self.epochs_data
            for epoch_data in epochs_to_show:
                epoch = epoch_data['epoch']
                train_acc = epoch_data['train_accuracy']
                test_acc = epoch_data.get('test_accuracy')
                epoch_time = epoch_data['epoch_time_seconds']
                
                test_acc_str = f"{test_acc:.2f}%" if test_acc is not None else "N/A"
                f.write(f"{epoch:<8} {train_acc:>10.2f}% {test_acc_str:>11} {epoch_time:>9.2f}\n")
            
            if len(self.epochs_data) > 10:
                f.write(f"\n(Showing last 10 of {len(self.epochs_data)} epochs)\n")
            
            f.write("\n" + "=" * 60 + "\n")
            
        print(f"Summary saved to {summary_path}")
        
    def get_model_path(self) -> str:
        """Get the path where the model should be saved."""
        return os.path.join(self.run_path, "model.pkl")
        
    def set_total_training_time(self, total_time: float):
        """Set the total training time."""
        self.total_training_time = total_time


def find_latest_run(
    base_dir: str = "models/training_runs",
    stage: Optional[str] = None,
    board_size: Optional[int] = None
) -> Optional[str]:
    """
    Find the most recent training run folder.
    
    Args:
        base_dir: Base directory for training runs
        stage: Filter by stage (None = any stage)
        board_size: Filter by board size (None = any size)
        
    Returns:
        Path to the most recent run folder, or None if not found
    """
    if not os.path.exists(base_dir):
        return None
        
    # Get all subdirectories
    run_folders = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    
    # Filter by stage and board size if specified
    if stage is not None:
        run_folders = [f for f in run_folders if f"_{stage}_" in os.path.basename(f)]
    if board_size is not None:
        run_folders = [f for f in run_folders if f"_{board_size}x{board_size}" in os.path.basename(f)]
    
    if not run_folders:
        return None
        
    # Return the most recent
    return max(run_folders, key=os.path.getctime)
