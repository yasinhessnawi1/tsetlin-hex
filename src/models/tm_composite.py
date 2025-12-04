"""
TM Composite - Ensemble of Specialized Graph Tsetlin Machines

Strategy 3 from GTM_Optimization_Guide.md:
Multiple specialized TMs with different configurations working together.

Expected clause reduction: 2-5x
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from .hex_graph_tm import HexGraphTM


@dataclass
class SpecialistConfig:
    """Configuration for a specialist TM."""
    name: str
    clauses: int
    depth: int
    s: float
    T: int
    message_size: int = 256
    message_bits: int = 2
    max_included_literals: int = 255
    weight: float = 1.0


class HexTMComposite:
    """
    Ensemble of specialized Graph Tsetlin Machines for Hex.

    Each specialist focuses on different aspects:
    - Shallow depth: Local patterns
    - Medium depth: Regional patterns
    - Deep depth: Global connectivity

    Or different specificities:
    - Low s: General patterns
    - Medium s: Balanced
    - High s: Specific patterns
    """

    def __init__(self, grid=(208, 1, 1), block=(128, 1, 1)):
        self.specialists: Dict[str, HexGraphTM] = {}
        self.configs: Dict[str, SpecialistConfig] = {}
        self.trained = False
        self.grid = grid
        self.block = block

    def add_specialist(self, config: SpecialistConfig):
        """Add a specialist with given configuration."""
        model = HexGraphTM(
            number_of_clauses=config.clauses,
            T=config.T,
            s=config.s,
            depth=config.depth,
            message_size=config.message_size,
            message_bits=config.message_bits,
            max_included_literals=config.max_included_literals,
            grid=self.grid,
            block=self.block
        )
        self.specialists[config.name] = model
        self.configs[config.name] = config
        print(f"[OK] Added specialist: {config.name}")
        print(f"     clauses={config.clauses}, depth={config.depth}, s={config.s}, weight={config.weight}")

    def total_clauses(self) -> int:
        """Get total clauses across all specialists."""
        return sum(c.clauses for c in self.configs.values())

    def fit(self, graphs, labels, epochs: int = 100):
        """Train all specialists."""
        print(f"\n{'='*60}")
        print(f"TRAINING TM COMPOSITE")
        print(f"{'='*60}")
        print(f"Specialists: {len(self.specialists)}")
        print(f"Total Clauses: {self.total_clauses()}")
        print(f"{'='*60}\n")

        for name, model in self.specialists.items():
            config = self.configs[name]
            print(f"\n--- Training Specialist: {name} ---")
            print(f"    Clauses: {config.clauses}, Depth: {config.depth}, s: {config.s}")
            model.fit(graphs, labels, epochs=epochs)
            print(f"[OK] {name} training complete")

        self.trained = True
        print(f"\n{'='*60}")
        print(f"COMPOSITE TRAINING COMPLETE")
        print(f"{'='*60}\n")

    def predict(self, graphs) -> np.ndarray:
        """Weighted voting prediction."""
        if not self.trained:
            raise ValueError("Composite not trained yet!")

        n_samples = graphs.number_of_graphs
        votes = np.zeros(n_samples, dtype=np.float32)

        for name, model in self.specialists.items():
            weight = self.configs[name].weight
            preds = model.predict(graphs)
            # Convert 0/1 to -1/+1 for voting
            votes += weight * (2 * preds - 1)

        # Final prediction: vote >= 0 means class 1
        return (votes >= 0).astype(np.int32)

    def predict_with_confidence(self, graphs):
        """Return predictions with confidence scores."""
        if not self.trained:
            raise ValueError("Composite not trained yet!")

        n_samples = graphs.number_of_graphs
        votes = np.zeros(n_samples, dtype=np.float32)
        total_weight = sum(c.weight for c in self.configs.values())

        for name, model in self.specialists.items():
            weight = self.configs[name].weight
            preds = model.predict(graphs)
            votes += weight * (2 * preds - 1)

        # Confidence is normalized absolute vote
        confidence = np.abs(votes) / total_weight
        predictions = (votes >= 0).astype(np.int32)

        return predictions, confidence

    def evaluate(self, graphs, labels) -> float:
        """Evaluate accuracy."""
        predictions = self.predict(graphs)
        accuracy = 100.0 * np.sum(predictions == labels) / len(labels)
        return accuracy

    def evaluate_specialists(self, graphs, labels) -> Dict[str, float]:
        """Evaluate each specialist individually."""
        if not self.trained:
            raise ValueError("Composite not trained yet!")

        results = {}
        for name, model in self.specialists.items():
            preds = model.predict(graphs)
            acc = 100.0 * np.sum(preds == labels) / len(labels)
            results[name] = acc

        return results

    def print_specialist_performance(self, graphs, labels):
        """Print detailed performance of each specialist."""
        print(f"\n{'='*60}")
        print("SPECIALIST PERFORMANCE")
        print(f"{'='*60}")

        results = self.evaluate_specialists(graphs, labels)
        composite_acc = self.evaluate(graphs, labels)

        print(f"\n{'Specialist':<20} {'Clauses':<10} {'Depth':<8} {'s':<8} {'Accuracy':<10}")
        print("-" * 60)

        for name in self.specialists.keys():
            config = self.configs[name]
            acc = results[name]
            print(f"{name:<20} {config.clauses:<10} {config.depth:<8} "
                  f"{config.s:<8.1f} {acc:<10.2f}%")

        print("-" * 60)
        print(f"{'COMPOSITE (weighted)':<46} {composite_acc:<10.2f}%")
        print(f"\nTotal Clauses: {self.total_clauses()}")
        print(f"{'='*60}\n")


# Predefined composite configurations

def create_depth_diverse_composite(base_clauses: int = 50, grid=(208, 1, 1), block=(128, 1, 1)) -> HexTMComposite:
    """
    Create composite with specialists at different depths.
    Good for capturing patterns at different scales.

    Updated: depth 1-4, weights 1.2 → 1.0 → 0.8 → 0.6
    """
    composite = HexTMComposite(grid=grid, block=block)

    configs = [
        SpecialistConfig("depth_1", clauses=base_clauses, depth=1, s=10.0, T=10000, weight=1.2),
        SpecialistConfig("depth_2", clauses=base_clauses, depth=2, s=10.0, T=10000, weight=1.0),
        SpecialistConfig("depth_3", clauses=base_clauses, depth=3, s=10.0, T=10000, weight=0.8),
        SpecialistConfig("depth_4", clauses=base_clauses, depth=4, s=10.0, T=10000, weight=0.6),
    ]

    for config in configs:
        composite.add_specialist(config)

    return composite


def create_specificity_diverse_composite(base_clauses: int = 50, grid=(208, 1, 1), block=(128, 1, 1)) -> HexTMComposite:
    """
    Create composite with specialists at different specificities.
    Good for capturing both general and specific patterns.
    """
    composite = HexTMComposite(grid=grid, block=block)

    configs = [
        SpecialistConfig("coarse", clauses=base_clauses, depth=3, s=5.0, T=15000, weight=0.8),
        SpecialistConfig("medium", clauses=base_clauses, depth=3, s=10.0, T=15000, weight=1.0),
        SpecialistConfig("fine", clauses=base_clauses, depth=3, s=15.0, T=15000, weight=1.0),
        SpecialistConfig("very_fine", clauses=base_clauses, depth=3, s=20.0, T=15000, weight=0.8),
    ]

    for config in configs:
        composite.add_specialist(config)

    return composite


def create_mixed_composite(base_clauses: int = 40, grid=(208, 1, 1), block=(128, 1, 1)) -> HexTMComposite:
    """
    Create composite with mixed depth and specificity diversity.
    Most robust approach.
    """
    composite = HexTMComposite(grid=grid, block=block)

    configs = [
        SpecialistConfig("shallow_general", clauses=base_clauses, depth=2, s=5.0, T=15000, weight=0.9),
        SpecialistConfig("medium_balanced", clauses=base_clauses, depth=3, s=10.0, T=15000, weight=1.0),
        SpecialistConfig("deep_balanced", clauses=base_clauses, depth=4, s=10.0, T=15000, weight=1.1),
        SpecialistConfig("medium_specific", clauses=base_clauses, depth=3, s=15.0, T=15000, weight=0.9),
        SpecialistConfig("deep_specific", clauses=base_clauses, depth=4, s=15.0, T=15000, weight=1.0),
    ]

    for config in configs:
        composite.add_specialist(config)

    return composite
