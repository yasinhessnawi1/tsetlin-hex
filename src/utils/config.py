"""
Configuration management for the Graph Tsetlin Machine Hex project.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """Configuration parameters for the project."""

    # Board settings
    board_size: int = 10

    # Data generation (A100 80GB optimized)
    num_train_games: int = 1000000  # Massive dataset for near-perfect accuracy
    num_test_games: int = 200000
    save_states_at: List[int] = None  # Will be set to [0, -2, -5] in __post_init__
    use_cuda: bool = True

    # Graph encoding
    hypervector_size: int = 128
    hypervector_bits: int = 4

    # GTM Model parameters - A100 80GB OPTIMIZED
    # Key insight: 100 clauses works best, but with massive data, can go to 200
    number_of_clauses: int = 100  # Scaled up for A100 + massive dataset
    T: int = 1500  # Optimal for binary classification
    s: float = 0.1  # Good balance for learning speed
    depth: int = 3  # Deep message passing for connectivity detection
    message_size: int = 256
    message_bits: int = 2
    max_included_literals: int = 255

    # CUDA configuration (for 6GB GPU)
    grid: tuple = (16*13, 1, 1)  # Optimized for A100-style
    block: tuple = (128, 1, 1)

    # Training (A100 80GB)
    epochs: int = 100  # Leverage massive dataset
    test_every: int = 5  # Check progress every 10 epochs

    # Paths
    data_dir: str = "data"
    models_dir: str = "models"

    def __post_init__(self):
        """Initialize default values for mutable types."""
        if self.save_states_at is None:
            self.save_states_at = [0, -2, -5]  # End, 2 before, 5 before

    def get_train_data_path(self) -> str:
        """Get path to training data."""
        return f"{self.data_dir}/train_games_{self.board_size}x{self.board_size}.npz"

    def get_test_data_path(self) -> str:
        """Get path to test data."""
        return f"{self.data_dir}/test_games_{self.board_size}x{self.board_size}.npz"

    def get_model_path(self, stage: str = "end") -> str:
        """Get path to save/load model."""
        return f"{self.models_dir}/gtm_{self.board_size}x{self.board_size}_{stage}.pkl"

    def print_config(self):
        """Print configuration summary."""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        print(f"\nBoard Settings:")
        print(f"  Board size: {self.board_size}x{self.board_size}")
        print(f"\nData Generation:")
        print(f"  Training games: {self.num_train_games}")
        print(f"  Testing games: {self.num_test_games}")
        print(f"  Save states at: {self.save_states_at}")
        print(f"  Use CUDA: {self.use_cuda}")
        print(f"\nGraph Encoding:")
        print(f"  Hypervector size: {self.hypervector_size}")
        print(f"  Hypervector bits: {self.hypervector_bits}")
        print(f"\nGTM Model:")
        print(f"  Number of clauses: {self.number_of_clauses}")
        print(f"  Threshold (T): {self.T}")
        print(f"  Specificity (s): {self.s}")
        print(f"  Message passing depth: {self.depth}")
        print(f"  Message size: {self.message_size}")
        print(f"  Message bits: {self.message_bits}")
        print(f"  Max included literals: {self.max_included_literals}")
        print(f"\nCUDA Configuration:")
        print(f"  Grid: {self.grid}")
        print(f"  Block: {self.block}")
        print(f"\nTraining:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Test every: {self.test_every} epochs")
        print(f"\nPaths:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Models directory: {self.models_dir}")
        print("="*60 + "\n")


# Default configuration instance
default_config = Config()


if __name__ == "__main__":
    # Test config
    config = Config()
    config.print_config()
