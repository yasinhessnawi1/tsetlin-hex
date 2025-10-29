"""
Quick test with smaller parameters to verify the fix works
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
from src.models import HexGraphTM, Predictor

def load_gtm_dataset(filepath: str):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    graphs = data['graphs']
    labels = data['labels']
    indices = data.get('indices', None)
    
    if indices is not None:
        print(f"Loaded {len(indices)} samples from {filepath} (using shared graphs)")
    else:
        print(f"Loaded {len(labels)} samples from {filepath}")
    
    return graphs, labels, indices

print("="*70, flush=True)
print("QUICK TEST: Verify Fix with Small Parameters", flush=True)
print("="*70, flush=True)

# Load datasets
train_graphs, train_labels, train_indices = load_gtm_dataset('data/train_gtm_5x5_end.pkl')
test_graphs, test_labels, test_indices = load_gtm_dataset('data/test_gtm_5x5_end.pkl')

print(f"\nTraining: {len(train_labels)}, Test: {len(test_labels)}")
print(f"Total graphs in object: {train_graphs.number_of_graphs}")

# SMALL PARAMETERS for quick test
print("\nUsing SMALL parameters for quick test:")
print("  Clauses: 500")
print("  T: 1000")
print("  s: 10")
print("  Depth: 2")
print("  Epochs: 10")

model = HexGraphTM(
    number_of_clauses=500,
    T=1000,
    s=10.0,
    depth=2,
    message_size=256,
    message_bits=2,
    max_included_literals=32,
    grid=(16*13, 1, 1),
    block=(128, 1, 1)
)

predictor = Predictor(model)

print("\nTraining for 10 epochs...")
print("NOTE: First epoch will take longer due to initialization")
print()

train_acc, test_acc = predictor.train(
    train_graphs=train_graphs,
    train_labels=train_labels,
    test_graphs=test_graphs,
    test_labels=test_labels,
    epochs=10,
    test_every=2
)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Training Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
print("="*70)

if test_acc > 60:
    print("\n[SUCCESS] GTM is learning! Test accuracy > 60%")
    print("The fix is working - train and test have compatible encodings!")
    print("\nYou can now run full training with benchmark parameters.")
else:
    print("\n[WARNING] Test accuracy still low.")
    print(f"But check if test_acc ({test_acc:.1f}%) > train_acc ({train_acc:.1f}%)")
    print("If test > train, the model might need more epochs or better hyperparameters.")

