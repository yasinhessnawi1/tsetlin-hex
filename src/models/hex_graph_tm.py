"""
Graph Tsetlin Machine wrapper for Hex winner prediction.
Configured for CUDA acceleration on 6GB GPU.
"""

import os
import numpy as np
import pickle
from typing import Tuple, Optional
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine


class HexGraphTM:
    """
    Wrapper for MultiClassGraphTsetlinMachine configured for Hex board games.
    """

    def __init__(
        self,
        number_of_clauses: int = 500,
        T: int = 15000,
        s: float = 10.0,
        depth: int = 3,
        message_size: int = 256,
        message_bits: int = 2,
        max_included_literals: int = 32,
        grid: Tuple[int, int, int] = (16*13, 1, 1),
        block: Tuple[int, int, int] = (128, 1, 1),
        seed: int = 42
    ):
        """
        Initialize the Graph Tsetlin Machine.

        Args:
            number_of_clauses: Number of clauses (pattern detectors)
            T: Threshold parameter (higher = more specific patterns)
            s: Specificity parameter (controls clause specialization)
            depth: Message passing depth (logical reasoning layers)
            message_size: Size of messages passed between nodes
            message_bits: Bits per message
            max_included_literals: Maximum literals per clause
            grid: CUDA grid dimensions
            block: CUDA block dimensions
            seed: Random seed for reproducibility
        """
        self.params = {
            'number_of_clauses': number_of_clauses,
            'T': T,
            's': s,
            'depth': depth,
            'message_size': message_size,
            'message_bits': message_bits,
            'max_included_literals': max_included_literals,
            'grid': grid,
            'block': block,
        }

        self.seed = seed
        self.tm = None
        self.trained = False

    def _create_tm(self):
        """Create the Tsetlin Machine instance."""
        # Check CUDA availability
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            device = pycuda.autoinit.device
            free_mem, total_mem = cuda.mem_get_info()
            print(f"\n✓ GPU Device: {device.name()}")
            print(f"  - Compute Capability: {device.compute_capability()}")
            print(f"  - Memory: {free_mem / (1024**3):.2f} GB free / {total_mem / (1024**3):.2f} GB total")
            print(f"  - Using CUDA for training\n")
        except Exception as e:
            print(f"\n⚠ WARNING: Could not access GPU: {e}")
            print("  Training will be very slow without GPU!\n")
        
        self.tm = MultiClassGraphTsetlinMachine(
            number_of_clauses=self.params['number_of_clauses'],
            T=self.params['T'],
            s=self.params['s'],
            depth=self.params['depth'],
            message_size=self.params['message_size'],
            message_bits=self.params['message_bits'],
            max_included_literals=self.params['max_included_literals'],
            grid=self.params['grid'],
            block=self.params['block'],
        )

    def fit(
        self,
        graphs: Graphs,
        labels: np.ndarray,
        epochs: int = 100,
        incremental: bool = False
    ):
        """
        Train the Graph Tsetlin Machine.

        Args:
            graphs: Training graphs
            labels: Training labels (0 or 1)
            epochs: Number of training epochs
            incremental: Whether to continue training existing model
        """
        if self.tm is None or not incremental:
            self._create_tm()

        print(f"\nTraining GTM for {epochs} epochs...")
        print(f"  Clauses: {self.params['number_of_clauses']}")
        print(f"  Depth: {self.params['depth']}")
        print(f"  T: {self.params['T']}, s: {self.params['s']}")

        for epoch in range(epochs):
            self.tm.fit(graphs, labels, epochs=1, incremental=True)
            self.trained = True

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                train_acc = self.evaluate(graphs, labels)
                print(f"Epoch {epoch + 1}/{epochs}: Train Accuracy = {train_acc:.2f}%")

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

        return self.tm.predict(graphs)

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

        predictions = self.tm.predict(graphs)
        accuracy = 100.0 * np.sum(predictions == labels) / len(labels)
        return accuracy

    def save(self, filepath: str):
        """
        Save the model to disk.
        
        NOTE: This saves the model's learned state using the GTM's save() method,
        which returns a dictionary containing all necessary state and config.
        We embed this in our own save file along with HexGraphTM metadata.
        
        Args:
            filepath: Path to save the model
        """
        if self.tm is None:
            raise ValueError("No model to save!")
        
        if not self.trained:
            print("WARNING: Saving an untrained model!")
        
        # Extract the GTM state dictionary
        # We use save() instead of get_state() because save() returns a dict
        # that can be passed to load(), which handles full initialization (kernels, etc.)
        try:
            tm_state = self.tm.save() # Returns dict, doesn't write to file if fname=""
        except AttributeError:
            print("WARNING: save() not available. Attempting to pickle entire object.")
            tm_state = None
            
        save_data = {
            'params': self.params,
            'seed': self.seed,
            'trained': self.trained,
            'tm_state': tm_state,
            'has_state': tm_state is not None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Verify file was written
        file_size = os.path.getsize(filepath)
        print(f"Model saved to {filepath} ({file_size / 1024:.2f} KB)")
        
        if file_size < 100:
            print("WARNING: Saved file is very small. This may indicate a serialization issue!")

    def load(self, filepath: str):
        """
        Load a model from disk.
        
        This reconstructs the GTM model from saved hyperparameters and learned state.
        The model is recreated and the state is restored using tm.load().
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.params = save_data['params']
        self.seed = save_data['seed']
        self.trained = save_data['trained']
        
        # Recreate the TM with saved hyperparameters
        self._create_tm()
        
        # Restore the learned state if available
        if save_data.get('has_state', False) and save_data.get('tm_state') is not None:
            try:
                # Use load() with state_dict to restore state AND initialize kernels
                self.tm.load(state_dict=save_data['tm_state'])
                print(f"Model loaded from {filepath} with learned state restored")
            except AttributeError:
                print("WARNING: load() not available. Model loaded but state not restored.")
                print("You may need to retrain the model.")
                self.trained = False
            except Exception as e:
                print(f"WARNING: Failed to load GTM state: {e}")
                self.trained = False
        else:
            print(f"Model loaded from {filepath} (hyperparameters only, no learned state)")
            print("You will need to retrain the model.")
            self.trained = False


    def print_info(self):
        """Print model information."""
        print("\n" + "="*60)
        print("Graph Tsetlin Machine Configuration")
        print("="*60)
        print(f"Number of clauses: {self.params['number_of_clauses']}")
        print(f"Threshold (T): {self.params['T']}")
        print(f"Specificity (s): {self.params['s']}")
        print(f"Message passing depth: {self.params['depth']}")
        print(f"Message size: {self.params['message_size']}")
        print(f"Message bits: {self.params['message_bits']}")
        print(f"Max included literals: {self.params['max_included_literals']}")
        print(f"\nCUDA Configuration:")
        print(f"Grid: {self.params['grid']}")
        print(f"Block: {self.params['block']}")
        print(f"\nStatus: {'Trained' if self.trained else 'Not trained'}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test the wrapper
    print("Testing HexGraphTM wrapper...")

    model = HexGraphTM(
        number_of_clauses=100,
        T=5000,
        depth=2
    )

    model.print_info()
