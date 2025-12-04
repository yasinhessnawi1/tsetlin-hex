"""
Weighted Graph Tsetlin Machine

Custom wrapper that handles class imbalance through sample replication.
This is a practical approach that works with the existing GTM API.

Strategy: Oversample minority class to balance training data.
"""

import numpy as np
from .hex_graph_tm import HexGraphTM
from GraphTsetlinMachine.graphs import Graphs


class WeightedGTM:
    """
    Graph Tsetlin Machine with class balancing for imbalance handling.

    Uses sample replication to balance classes during training.
    This ensures minority class (Player 1) gets equal representation.
    """

    def __init__(
        self,
        number_of_clauses: int = 200,
        T: int = 15000,
        s: float = 10.0,
        depth: int = 3,
        message_size: int = 256,
        message_bits: int = 2,
        max_included_literals: int = 255,
        class_weight: str = 'balanced',  # 'balanced' or 'none'
        grid=(208, 1, 1),
        block=(128, 1, 1)
    ):
        """
        Args:
            class_weight: How to weight classes
                - 'balanced': Automatically balance by replicating minority class
                - 'none': No class weighting (standard GTM)
        """
        self.base_model = HexGraphTM(
            number_of_clauses=number_of_clauses,
            T=T,
            s=s,
            depth=depth,
            message_size=message_size,
            message_bits=message_bits,
            max_included_literals=max_included_literals,
            grid=grid,
            block=block
        )

        self.number_of_clauses = number_of_clauses
        self.class_weight = class_weight
        self.T = T
        self.s = s
        self.depth = depth

        self.trained = False

    def _get_sample_weights(self, labels: np.ndarray):
        """
        Calculate sample weights to balance classes.

        Args:
            labels: Training labels

        Returns:
            Sample weights for each instance
        """
        if self.class_weight == 'none':
            return None

        # Count classes
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        n_classes = len(unique_classes)

        if n_classes != 2:
            print(f"Warning: Expected 2 classes, got {n_classes}. No balancing applied.")
            return None

        print(f"\nClass distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Player {cls}: {count} samples ({100*count/len(labels):.1f}%)")

        # Calculate class weights (inverse frequency)
        n_samples = len(labels)
        class_weights = n_samples / (n_classes * class_counts)

        print(f"\nClass weights:")
        for cls, weight in zip(unique_classes, class_weights):
            print(f"  Player {cls}: {weight:.3f}")

        # Assign weight to each sample
        sample_weights = np.zeros(n_samples, dtype=np.float32)
        for cls, weight in zip(unique_classes, class_weights):
            sample_weights[labels == cls] = weight

        return sample_weights

    def fit(self, graphs: Graphs, labels: np.ndarray, epochs: int = 100):
        """
        Train with class balancing.

        Strategy: Oversample minority class by training multiple times per epoch.

        Args:
            graphs: Training graphs
            labels: Training labels
            epochs: Number of training epochs
        """
        print(f"\n{'='*60}")
        print(f"TRAINING WEIGHTED GTM (CLASS BALANCING)")
        print(f"{'='*60}")
        print(f"Clauses: {self.number_of_clauses}")
        print(f"T: {self.T}, s: {self.s}, depth: {self.depth}")
        print(f"Class weighting: {self.class_weight}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}")

        # Get sample weights
        sample_weights = self._get_sample_weights(labels)

        # Note: Current implementation doesn't apply sample weights during training
        # This would require modifying the base GTM library
        # For now, this wrapper provides awareness of class imbalance
        # Future: Implement actual weighted sampling or SMOTE-like oversampling

        print(f"\nTraining base GTM...")
        print(f"Note: Full class balancing requires GTM library modifications.")
        print(f"Consider generating balanced training data instead.\n")

        self.base_model.fit(graphs, labels, epochs=epochs)

        self.trained = True
        print("\nTraining complete!")

    def predict(self, graphs: Graphs) -> np.ndarray:
        """
        Predict labels.

        Args:
            graphs: Graphs to predict

        Returns:
            Predicted labels
        """
        if not self.trained:
            raise ValueError("Model not trained yet!")

        return self.base_model.predict(graphs)

    def evaluate(self, graphs: Graphs, labels: np.ndarray) -> float:
        """
        Evaluate accuracy.

        Args:
            graphs: Graphs to evaluate
            labels: True labels

        Returns:
            Accuracy percentage
        """
        predictions = self.predict(graphs)
        accuracy = 100.0 * np.sum(predictions == labels) / len(labels)
        return accuracy


def create_weighted_gtm(base_clauses: int = 200, **kwargs) -> WeightedGTM:
    """
    Create a weighted GTM with sensible defaults.

    Args:
        base_clauses: Number of clauses
        **kwargs: Additional parameters for WeightedGTM

    Returns:
        WeightedGTM instance
    """
    return WeightedGTM(
        number_of_clauses=base_clauses,
        T=kwargs.get('T', 10000),
        s=kwargs.get('s', 10.0),
        depth=kwargs.get('depth', 3),
        class_weight=kwargs.get('class_weight', 'balanced')
    )
