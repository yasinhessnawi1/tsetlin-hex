"""
Predictor module for training and evaluating Hex winner prediction models.
"""

import numpy as np
import pickle
import time
from typing import Dict, Tuple
from GraphTsetlinMachine.graphs import Graphs

from .hex_graph_tm import HexGraphTM


class Predictor:
    """
    Manages training and evaluation of GTM models for Hex winner prediction.
    """

    def __init__(self, model: HexGraphTM, logger=None):
        """
        Initialize the predictor.

        Args:
            model: HexGraphTM instance
            logger: Optional TrainingLogger instance for structured logging
        """
        self.model = model
        self.logger = logger
        self.training_history = []

    def train(
        self,
        train_graphs: Graphs,
        train_labels: np.ndarray,
        test_graphs: Graphs,
        test_labels: np.ndarray,
        epochs: int = 100,
        test_every: int = 10
    ):
        """
        Train the model with periodic evaluation.

        Args:
            train_graphs: Training graphs (can be shared with test_graphs)
            train_labels: Training labels
            test_graphs: Test graphs (can be shared with train_graphs)
            test_labels: Test labels
            epochs: Number of training epochs
            test_every: Evaluate on test set every N epochs
            train_indices: Optional indices to subset train_graphs
            test_indices: Optional indices to subset test_graphs
        """
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        self.model.print_info()

        print(f"Training samples: {len(train_labels)}")
        print(f"Test samples: {len(test_labels)}")
        print(f"Class distribution (train): Player 0: {np.sum(train_labels == 0)}, Player 1: {np.sum(train_labels == 1)}")
        print()

        start_time = time.time()

        # Create the TM on first call
        if self.model.tm is None:
            self.model._create_tm()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train for one epoch
            self.model.tm.fit(train_graphs, train_labels, epochs=1, incremental=(epoch > 0))
            self.model.trained = True

            # Evaluate
            train_acc = self.model.evaluate(train_graphs, train_labels)

            epoch_time = time.time() - epoch_start

            # Test periodically
            if (epoch + 1) % test_every == 0 or epoch == 0 or epoch == epochs - 1:
                test_acc = self.model.evaluate(test_graphs, test_labels)

                # Record history
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'time': epoch_time
                })
                
                # Log to TrainingLogger if available
                if self.logger is not None:
                    self.logger.log_epoch(epoch + 1, train_acc, test_acc, epoch_time)

                print(f"Epoch {epoch + 1:3d}/{epochs}: "
                      f"Train = {train_acc:6.2f}%, "
                      f"Test = {test_acc:6.2f}%, "
                      f"Time = {epoch_time:.2f}s")
            else:
                # Record history (without test)
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_acc': train_acc,
                    'test_acc': None,
                    'time': epoch_time
                })
                
                # Log to TrainingLogger if available
                if self.logger is not None:
                    self.logger.log_epoch(epoch + 1, train_acc, None, epoch_time)

                print(f"Epoch {epoch + 1:3d}/{epochs}: "
                      f"Train = {train_acc:6.2f}%, "
                      f"Time = {epoch_time:.2f}s")

        total_time = time.time() - start_time
        
        # Set total training time in logger
        if self.logger is not None:
            self.logger.set_total_training_time(total_time)

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per epoch: {total_time/epochs:.2f}s")

        # Final evaluation
        final_train_acc = self.model.evaluate(train_graphs, train_labels)
        final_test_acc = self.model.evaluate(test_graphs, test_labels)

        print(f"\nFinal Training Accuracy: {final_train_acc:.2f}%")
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")
        print("="*60 + "\n")

        return final_train_acc, final_test_acc

    def evaluate_detailed(
        self,
        graphs: Graphs,
        labels: np.ndarray,
        name: str = "Dataset"
    ) -> Dict:
        """
        Perform detailed evaluation.

        Args:
            graphs: Graphs to evaluate
            labels: True labels
            name: Name of the dataset

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"DETAILED EVALUATION: {name}")
        print(f"{'='*60}")

        predictions = self.model.predict(graphs)

        # Overall accuracy
        accuracy = 100.0 * np.sum(predictions == labels) / len(labels)

        # Per-class accuracy
        player0_mask = labels == 0
        player1_mask = labels == 1

        player0_correct = np.sum(predictions[player0_mask] == 0)
        player1_correct = np.sum(predictions[player1_mask] == 1)

        player0_acc = 100.0 * player0_correct / np.sum(player0_mask) if np.sum(player0_mask) > 0 else 0
        player1_acc = 100.0 * player1_correct / np.sum(player1_mask) if np.sum(player1_mask) > 0 else 0

        # Confusion matrix
        tp = np.sum((predictions == 1) & (labels == 1))  # True positives
        tn = np.sum((predictions == 0) & (labels == 0))  # True negatives
        fp = np.sum((predictions == 1) & (labels == 0))  # False positives
        fn = np.sum((predictions == 0) & (labels == 1))  # False negatives

        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        print(f"\nPer-class Accuracy:")
        print(f"  Player 0: {player0_acc:.2f}% ({player0_correct}/{np.sum(player0_mask)})")
        print(f"  Player 1: {player1_acc:.2f}% ({player1_correct}/{np.sum(player1_mask)})")

        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              P0      P1")
        print(f"Actual  P0   {tn:5d}   {fp:5d}")
        print(f"        P1   {fn:5d}   {tp:5d}")

        print(f"{'='*60}\n")

        results = {
            'name': name,
            'accuracy': accuracy,
            'player0_accuracy': player0_acc,
            'player1_accuracy': player1_acc,
            'confusion_matrix': {
                'tn': tn, 'fp': fp,
                'fn': fn, 'tp': tp
            },
            'predictions': predictions,
            'labels': labels
        }

        return results

    def save_training_history(self, filepath: str):
        """Save training history to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.training_history, f)
        print(f"Training history saved to {filepath}")

    def load_training_history(self, filepath: str):
        """Load training history from disk."""
        with open(filepath, 'rb') as f:
            self.training_history = pickle.load(f)
        print(f"Training history loaded from {filepath}")


if __name__ == "__main__":
    # Test the predictor
    print("Testing Predictor module...")

    from src.models import HexGraphTM

    model = HexGraphTM(
        number_of_clauses=100,
        T=5000,
        depth=2
    )

    predictor = Predictor(model)
    print("Predictor initialized successfully!")
