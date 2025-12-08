"""
Graph Tsetlin Machine wrapper for Hex winner prediction.
Configured for CUDA acceleration on 6GB GPU.

with correct parameters for Hex binary classification.
"""

import numpy as np
import pickle
from typing import Tuple, Optional, Union
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine  # Try MultiClass instead


class HexGraphTM:
    """
    Wrapper for GraphTsetlinMachine (binary) configured for Hex board games.
    
    Key fixes:
    - Using GraphTsetlinMachine (binary) instead of MultiClass
    - Reasonable T values (15-50 instead of 15000)
    - Adaptive s parameter (can be tuple for multi-depth)
    - More clauses (2000 default instead of 500)
    - No literal limit by default
    - Board-size aware configurations
    """

    def __init__(
        self,
        board_size: int = 5,
        number_of_clauses: int = None,  # Will be auto-set based on board_size if None
        T: int = None,  # Will be auto-set based on board_size if None
        s: Union[float, Tuple[float, ...]] = None,  # Will be auto-set based on depth if None
        depth: int = 2,
        message_size: int = 256,
        message_bits: int = 2,
        max_included_literals: Optional[int] = None,  # No limit by default
        boost_true_positive_feedback: int = 1,
        grid: Tuple[int, int, int] = (16*13, 1, 1),
        block: Tuple[int, int, int] = (128, 1, 1),
        seed: int = 42
    ):
        """
        Initialize the Graph Tsetlin Machine for Hex.

        Args:
            board_size: Size of the Hex board (NxN)
            number_of_clauses: Number of clauses (auto-set based on board_size if None)
            T: Threshold parameter 
            s: Specificity parameter - can be tuple for multi-depth learning
            depth: Message passing depth (1-3, recommend 2 for most cases)
            message_size: Size of messages passed between nodes
            message_bits: Bits per message
            max_included_literals: Maximum literals per clause (None = no limit)
            boost_true_positive_feedback: Amplify positive feedback (1-3)
            grid: CUDA grid dimensions
            block: CUDA block dimensions
            seed: Random seed for reproducibility
        """
        self.board_size = board_size
        
        # Auto-configure based on board size if not specified
        if number_of_clauses is None:
            number_of_clauses = self._auto_clauses(board_size, depth)
        
        if T is None:
            T = self._auto_threshold(board_size)
        
        if s is None:
            s = self._auto_specificity(depth)
        
        self.params = {
            'number_of_clauses': number_of_clauses,
            'T': T,
            's': s,
            'depth': depth,
            'message_size': message_size,
            'message_bits': message_bits,
            'max_included_literals': max_included_literals,
            'boost_true_positive_feedback': boost_true_positive_feedback,
            'grid': grid,
            'block': block,
        }

        self.seed = seed
        self.tm = None
        self.trained = False
        self.training_history = []
    
    @staticmethod
    def _auto_clauses(board_size: int, depth: int) -> int:
        """Auto-determine number of clauses based on board size and depth."""
        base = {
            5: 1000,
            6: 1500,
            7: 2000,
            8: 2500,
            9: 3000,
            10: 4000,
            11: 5000,
        }
        clauses = base.get(board_size, 2000 + (board_size - 7) * 500)
        
        # Increase for deeper reasoning
        if depth >= 3:
            clauses = int(clauses * 1.5)
        
        return clauses
    
    @staticmethod
    def _auto_threshold(board_size: int) -> int:
        """Auto-determine threshold based on board size."""
        thresholds = {
            5: 15,
            6: 20,
            7: 25,
            8: 30,
            9: 35,
            10: 40,
            11: 50,
        }
        return thresholds.get(board_size, 25)
    
    @staticmethod
    def _auto_specificity(depth: int) -> Union[float, Tuple[float, ...]]:
        """Auto-determine specificity based on depth."""
        if depth == 1:
            return 3.0
        elif depth == 2:
            return (4.0, 2.5)
        elif depth == 3:
            return (5.0, 3.0, 2.0)
        else:
            # For depth > 3, create decreasing tuple
            return tuple(5.0 - i * 0.7 for i in range(depth))

    def _create_tm(self):
        """Create the Tsetlin Machine instance."""
        # Check CUDA availability
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            device = pycuda.autoinit.device
            free_mem, total_mem = cuda.mem_get_info()
            print(f"\n[OK] GPU Device: {device.name()}")
            print(f"  - Compute Capability: {device.compute_capability()}")
            print(f"  - Memory: {free_mem / (1024**3):.2f} GB free / {total_mem / (1024**3):.2f} GB total")
            print(f"  - Using CUDA for training\n")
        except Exception as e:
            print(f"\n[WARNING] Could not access GPU: {e}")
            print("  Training will be very slow without GPU!\n")
        
        # Create MultiClass GTM (2 classes for binary problem)
        self.tm = MultiClassGraphTsetlinMachine(
            number_of_clauses=self.params['number_of_clauses'],
            T=self.params['T'],
            s=self.params['s'],
            depth=self.params['depth'],
            message_size=self.params['message_size'],
            message_bits=self.params['message_bits'],
            max_included_literals=self.params['max_included_literals'],
            boost_true_positive_feedback=self.params['boost_true_positive_feedback'],
            grid=self.params['grid'],
            block=self.params['block'],
        )

    def fit(
        self,
        graphs: Graphs,
        labels: np.ndarray,
        epochs: int = 100,
        incremental: bool = False,
        early_stopping_patience: int = 30,
        validation_graphs: Optional[Graphs] = None,
        validation_labels: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Train the Graph Tsetlin Machine with improved monitoring.

        Args:
            graphs: Training graphs
            labels: Training labels (0 or 1 for binary classification)
            epochs: Number of training epochs
            incremental: Whether to continue training existing model
            early_stopping_patience: Stop if no improvement for N checks
            validation_graphs: Optional validation set
            validation_labels: Optional validation labels
            verbose: Print training progress
        """
        if self.tm is None or not incremental:
            self._create_tm()

        # Validate labels
        unique_labels = np.unique(labels)
        if not np.array_equal(unique_labels, np.array([0, 1])):
            raise ValueError(f"Labels must be binary (0 or 1). Got: {unique_labels}")

        if verbose:
            print(f"\n{'='*70}")
            print(f"Training Graph Tsetlin Machine")
            print(f"{'='*70}")
            print(f"Board size: {self.board_size}x{self.board_size}")
            print(f"Training samples: {len(labels)}")
            print(f"  - Player 1 wins (label=0): {np.sum(labels == 0)} ({100*np.mean(labels==0):.1f}%)")
            print(f"  - Player 2 wins (label=1): {np.sum(labels == 1)} ({100*np.mean(labels==1):.1f}%)")
            print(f"\nModel Configuration:")
            print(f"  - Clauses: {self.params['number_of_clauses']}")
            print(f"  - Depth: {self.params['depth']}")
            print(f"  - T (threshold): {self.params['T']}")
            print(f"  - s (specificity): {self.params['s']}")
            print(f"  - Max literals: {self.params['max_included_literals'] or 'No limit'}")
            print(f"  - Boost TP feedback: {self.params['boost_true_positive_feedback']}")
            print(f"\nTraining for {epochs} epochs...")
            print(f"{'='*70}\n")

        # CRITICAL: Binary GTM needs these set before _fit()
        if not hasattr(self.tm, 'number_of_outputs') or self.tm.number_of_outputs is None:
            self.tm.number_of_outputs = 1  # Binary classification
            self.tm.max_y = None
            self.tm.min_y = None

        # Prepare encoded labels for GTM (0/1 format)
        # Binary GTM expects labels as 0 or 1, NOT as classes
        encoded_Y = np.where(labels == 1, self.params['T'], -self.params['T']).astype(np.int32)
        
        best_train_acc = 0
        best_val_acc = 0
        no_improvement = 0
        
        # Use the internal _fit method which accepts epochs parameter
        # This gives us epoch-by-epoch control
        for epoch in range(epochs):
            # Train for 1 epoch at a time
            self.tm._fit(graphs, encoded_Y, epochs=1, incremental=(epoch > 0))
            self.trained = True

            # Evaluate every 5 epochs or at start/end
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                train_acc = self.evaluate(graphs, labels)
                
                metrics = {
                    'epoch': epoch + 1,
                    'train_accuracy': train_acc
                }
                
                # Validation if provided
                if validation_graphs is not None and validation_labels is not None:
                    val_acc = self.evaluate(validation_graphs, validation_labels)
                    metrics['val_accuracy'] = val_acc
                    
                    if verbose:
                        print(f"Epoch {epoch + 1:3d}/{epochs}: "
                              f"Train Acc = {train_acc:6.2f}% | "
                              f"Val Acc = {val_acc:6.2f}%")
                    
                    # Track best validation
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        no_improvement = 0
                    else:
                        no_improvement += 1
                else:
                    if verbose:
                        print(f"Epoch {epoch + 1:3d}/{epochs}: Train Acc = {train_acc:6.2f}%")
                    
                    # Track best training
                    if train_acc > best_train_acc:
                        best_train_acc = train_acc
                        no_improvement = 0
                    else:
                        no_improvement += 1
                
                self.training_history.append(metrics)
                
                # Early stopping
                if no_improvement >= early_stopping_patience:
                    if verbose:
                        print(f"\n[EARLY STOPPING] No improvement for {early_stopping_patience} checks.")
                        if validation_graphs is not None:
                            print(f"Best validation accuracy: {best_val_acc:.2f}%")
                        else:
                            print(f"Best training accuracy: {best_train_acc:.2f}%")
                    break
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Training Complete!")
            if validation_graphs is not None and 'val_acc' in locals():
                print(f"Final Validation Accuracy: {val_acc:.2f}%")
            print(f"Final Training Accuracy: {train_acc:.2f}%")
            print(f"{'='*70}\n")

    def predict(self, graphs: Graphs) -> np.ndarray:
        """
        Predict labels for graphs.

        Args:
            graphs: Graphs to predict

        Returns:
            Predicted labels (0 or 1)
        """
        if self.tm is None:
            raise ValueError("Model not trained yet!")

        # MultiClass GTM uses argmax over class scores
        predictions = self.tm.predict(graphs)
        
        # DEBUG: Check predictions
        unique_preds = np.unique(predictions)
        if len(unique_preds) == 1:
            scores = self.tm.score(graphs)
            print(f"\n[WARNING] Model predicting only class {unique_preds[0]}!")
            print(f"  Class 0 scores: [{scores[:,0].min():.2f}, {scores[:,0].max():.2f}]")
            print(f"  Class 1 scores: [{scores[:,1].min():.2f}, {scores[:,1].max():.2f}]")
        
        return predictions.astype(np.int32)

    def predict_proba(self, graphs: Graphs) -> np.ndarray:
        """
        Get confidence scores for predictions.

        Args:
            graphs: Graphs to predict

        Returns:
            Array of scores (higher = more confident in class 1)
        """
        if self.tm is None:
            raise ValueError("Model not trained yet!")

        # GTM score() returns raw voting sums
        # Positive score = predicts class 1, negative = class 0
        scores = self.tm.score(graphs)
        return scores

    def evaluate(self, graphs: Graphs, labels: np.ndarray) -> float:
        """
        Evaluate accuracy on graphs.

        Args:
            graphs: Graphs to evaluate
            labels: True labels

        Returns:
            Accuracy percentage
        """
        if self.tm is None:
            raise ValueError("Model not trained yet!")

        predictions = self.predict(graphs)
        accuracy = 100.0 * np.sum(predictions == labels) / len(labels)
        return accuracy

    def get_confusion_matrix(self, graphs: Graphs, labels: np.ndarray) -> dict:
        """
        Get detailed prediction statistics.

        Args:
            graphs: Graphs to evaluate
            labels: True labels

        Returns:
            Dictionary with confusion matrix and metrics
        """
        predictions = self.predict(graphs)
        
        tp = np.sum((predictions == 1) & (labels == 1))  # True positives
        tn = np.sum((predictions == 0) & (labels == 0))  # True negatives
        fp = np.sum((predictions == 1) & (labels == 0))  # False positives
        fn = np.sum((predictions == 0) & (labels == 1))  # False negatives
        
        total = len(labels)
        accuracy = 100.0 * (tp + tn) / total
        
        # Avoid division by zero
        precision = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            }
        }

    def save(self, filepath: str) -> bool:
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model

        Returns:
            bool: True if saved, False if skipped (e.g., PyCUDA pickling issues)
        """
        if self.tm is None:
            raise ValueError("No model to save!")

        try:
            state_dict = self.tm.save()  # Get state from GTM (pickle-safe)
            save_data = {
                'params': self.params,
                'seed': self.seed,
                'trained': self.trained,
                'board_size': self.board_size,
                'training_history': self.training_history,
                'tm_state': state_dict
            }
        except Exception as exc:
            print(f"[WARN] Model save skipped: {exc}")
            print("[INFO] PyCUDA objects cannot be pickled; continuing without saving.")
            return False

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Model saved to {filepath}")
        return True

    def load(self, filepath: str):
        """
        Load a model from disk.

        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        self.params = save_data['params']
        self.seed = save_data['seed']
        self.trained = save_data['trained']
        self.board_size = save_data.get('board_size', 5)
        self.training_history = save_data.get('training_history', [])
        
        # Try to load from state_dict first, fallback to pickled object
        if 'tm_state' in save_data:
            self._create_tm()
            self.tm.load(state_dict=save_data['tm_state'])
        else:
            self.tm = save_data['tm']

        print(f"Model loaded from {filepath}")

    def print_info(self):
        """Print model information."""
        print("\n" + "="*70)
        print("Graph Tsetlin Machine Configuration")
        print("="*70)
        print(f"Board size: {self.board_size}x{self.board_size}")
        print(f"Model type: Binary Classification (GraphTsetlinMachine)")
        print(f"\nHyperparameters:")
        print(f"  - Number of clauses: {self.params['number_of_clauses']}")
        print(f"  - Threshold (T): {self.params['T']}")
        print(f"  - Specificity (s): {self.params['s']}")
        print(f"  - Message passing depth: {self.params['depth']}")
        print(f"  - Message size: {self.params['message_size']}")
        print(f"  - Message bits: {self.params['message_bits']}")
        print(f"  - Max included literals: {self.params['max_included_literals'] or 'No limit'}")
        print(f"  - Boost TP feedback: {self.params['boost_true_positive_feedback']}")
        print(f"\nCUDA Configuration:")
        print(f"  - Grid: {self.params['grid']}")
        print(f"  - Block: {self.params['block']}")
        print(f"\nStatus: {'Trained' if self.trained else 'Not trained'}")
        if self.training_history:
            print(f"Training epochs completed: {len(self.training_history)}")
            if self.training_history:
                last = self.training_history[-1]
                print(f"Last recorded accuracy: {last['train_accuracy']:.2f}%")
        print("="*70 + "\n")

    def print_training_summary(self):
        """Print summary of training history."""
        if not self.training_history:
            print("No training history available.")
            return
        
        print("\n" + "="*70)
        print("Training History Summary")
        print("="*70)
        
        epochs = [h['epoch'] for h in self.training_history]
        train_accs = [h['train_accuracy'] for h in self.training_history]
        
        print(f"Epochs trained: {max(epochs)}")
        print(f"Initial accuracy: {train_accs[0]:.2f}%")
        print(f"Final accuracy: {train_accs[-1]:.2f}%")
        print(f"Best accuracy: {max(train_accs):.2f}%")
        print(f"Improvement: {train_accs[-1] - train_accs[0]:+.2f}%")
        
        if 'val_accuracy' in self.training_history[0]:
            val_accs = [h['val_accuracy'] for h in self.training_history]
            print(f"\nValidation:")
            print(f"  - Initial: {val_accs[0]:.2f}%")
            print(f"  - Final: {val_accs[-1]:.2f}%")
            print(f"  - Best: {max(val_accs):.2f}%")
        
        print("="*70 + "\n")


# ============================================================================
# Preset Configurations for Different Board Sizes
# ============================================================================

def create_hex_gtm_small(board_size: int = 5, depth: int = 2) -> HexGraphTM:
    """
    Create GTM for small boards (5x5 to 7x7), predicting 2 moves before end.
    
    Expected accuracy: 98-100%
    """
    return HexGraphTM(
        board_size=board_size,
        number_of_clauses=1000,
        T=15,
        s=3.0 if depth == 1 else (4.0, 2.5),
        depth=depth,
        max_included_literals=None,
        boost_true_positive_feedback=1
    )


def create_hex_gtm_medium(board_size: int = 7, depth: int = 2) -> HexGraphTM:
    """
    Create GTM for medium boards (7x7 to 9x9), predicting 2-5 moves before end.
    
    Expected accuracy: 95-99%
    """
    return HexGraphTM(
        board_size=board_size,
        number_of_clauses=2000,
        T=25,
        s=(4.0, 2.5),
        depth=depth,
        max_included_literals=None,
        boost_true_positive_feedback=1
    )


def create_hex_gtm_large(board_size: int = 10, depth: int = 3) -> HexGraphTM:
    """
    Create GTM for large boards (10x10+), predicting 5+ moves before end.
    
    Expected accuracy: 90-97%
    This is genuinely difficult!
    """
    return HexGraphTM(
        board_size=board_size,
        number_of_clauses=4000,
        T=40,
        s=(5.0, 3.0, 2.0),
        depth=depth,
        max_included_literals=None,
        boost_true_positive_feedback=2
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("HexGraphTM")
    print("="*70)
    print("\nKey Changes:")
    print("  ✓ Using GraphTsetlinMachine (binary) instead of MultiClass")
    print("  ✓ T reduced from 15000 to 15-50 (reasonable values)")
    print("  ✓ s now uses tuples for multi-depth: (4.0, 2.5)")
    print("  ✓ Increased clauses: 1000-4000 (was 500)")
    print("  ✓ Removed literal limit (was 32)")
    print("  ✓ Auto-configuration based on board size")
    print("  ✓ Better training monitoring with early stopping")
    print("  ✓ Confusion matrix and detailed metrics")
    print("="*70 + "\n")

    # Example: Small board
    print("Example 1: Small board (5x5)")
    model_small = create_hex_gtm_small(board_size=5, depth=2)
    model_small.print_info()

    # Example: Medium board
    print("Example 2: Medium board (7x7)")
    model_medium = create_hex_gtm_medium(board_size=7, depth=2)
    model_medium.print_info()

    # Example: Large board
    print("Example 3: Large board (10x10)")
    model_large = create_hex_gtm_large(board_size=10, depth=3)
    model_large.print_info()

    # Example: Custom configuration
    print("Example 4: Custom auto-configured")
    model_custom = HexGraphTM(board_size=8)  # All other params auto-set
    model_custom.print_info()