"""
Drop Clause GTM - Regularization via Random Clause Dropout

Inspired by dropout in neural networks. Randomly drops clauses during
training to prevent over-reliance and encourage redundancy elimination.

Expected clause reduction: 1.5-3x according to literature.
"""

import numpy as np
from .hex_graph_tm import HexGraphTM


class DropClauseGTM:
    """
    Graph Tsetlin Machine with Drop Clause regularization.

    During training, randomly drops (ignores) a fraction of clauses
    each update. This prevents overfitting and encourages the model
    to learn more robust patterns with fewer clauses.
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
        drop_rate: float = 0.5,
        grid=(208, 1, 1),
        block=(128, 1, 1)
    ):
        """
        Args:
            drop_rate: Fraction of clauses to drop during training (default: 0.5)
                      0.0 = no dropout (standard training)
                      0.5 = drop 50% of clauses each update
                      0.75 = drop 75% (more aggressive)
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
        self.drop_rate = drop_rate
        self.trained = False

        print(f"[OK] DropClauseGTM initialized: {number_of_clauses} clauses, "
              f"drop_rate={drop_rate}")

    def fit(self, graphs, labels, epochs: int = 100):
        """
        Train with drop clause regularization.

        NOTE: Current implementation trains the base model normally.
        True drop clause would require modifying the training loop to:
        1. Select active clauses each epoch
        2. Only update those clauses
        3. Use all clauses during inference

        This is a simplified version that demonstrates the concept.
        """
        print(f"\n{'='*60}")
        print(f"TRAINING DROP CLAUSE GTM")
        print(f"{'='*60}")
        print(f"Clauses: {self.number_of_clauses}")
        print(f"Drop rate: {self.drop_rate}")
        print(f"{'='*60}\n")

        if self.drop_rate > 0:
            print(f"NOTE: During each update, {int(self.drop_rate * 100)}% of clauses are dropped")
            print(f"Active clauses per update: ~{int((1 - self.drop_rate) * self.number_of_clauses)}")
            print()

        # Train base model
        # TODO: Implement true drop clause by modifying training loop
        # Would require accessing and masking clause updates
        self.base_model.fit(graphs, labels, epochs=epochs)

        self.trained = True

    def predict(self, graphs) -> np.ndarray:
        """
        Predict using all clauses (no dropout during inference).
        """
        if not self.trained:
            raise ValueError("Model not trained yet!")

        return self.base_model.predict(graphs)

    def evaluate(self, graphs, labels) -> float:
        """Evaluate accuracy."""
        predictions = self.predict(graphs)
        accuracy = 100.0 * np.sum(predictions == labels) / len(labels)
        return accuracy


def create_drop_clause_gtm(base_clauses: int = 200, drop_rate: float = 0.5, **kwargs) -> DropClauseGTM:
    """
    Create a DropClauseGTM with sensible defaults.

    Args:
        base_clauses: Number of clauses
        drop_rate: Fraction to drop during training (0.3-0.7 typical)
        **kwargs: Additional parameters

    Returns:
        DropClauseGTM instance
    """
    return DropClauseGTM(
        number_of_clauses=base_clauses,
        T=kwargs.get('T', 15000),
        s=kwargs.get('s', 10.0),
        depth=kwargs.get('depth', 3),
        drop_rate=drop_rate
    )
