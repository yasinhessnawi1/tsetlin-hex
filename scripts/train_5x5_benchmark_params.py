"""
Train 5x5 with BENCHMARK parameters (Clauses=10000, T=8000, s=100)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
from src.models import HexGraphTM, Predictor

def load_gtm_dataset(filepath: str):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data['labels'])} samples from {filepath}")
    return data['graphs'], data['labels']

print("="*60)
print("TRAINING 5x5 WITH BENCHMARK PARAMETERS")
print("="*60)

# Load datasets
train_graphs, train_labels = load_gtm_dataset('data/train_gtm_5x5_end.pkl')
test_graphs, test_labels = load_gtm_dataset('data/test_gtm_5x5_end.pkl')

print(f"\nTraining: {len(train_labels)}, Test: {len(test_labels)}")

# BENCHMARK PARAMETERS from the image
print("\nUsing BENCHMARK parameters:")
print("  Clauses: 10000")
print("  T: 8000")
print("  s: 100")
print("  Depth: 2")
print("  Max weight: 255")

model = HexGraphTM(
    number_of_clauses=10000,
    T=8000,
    s=100.0,
    depth=2,
    message_size=256,
    message_bits=2,
    max_included_literals=255,
    grid=(16*13, 1, 1),
    block=(128, 1, 1)
)

predictor = Predictor(model)

print("\nTraining for 50 epochs...")
train_acc, test_acc = predictor.train(
    train_graphs=train_graphs,
    train_labels=train_labels,
    test_graphs=test_graphs,
    test_labels=test_labels,
    epochs=50,
    test_every=5
)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Training Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
print("="*60)

if test_acc > 85:
    print("✅ EXCELLENT! Matches benchmark performance (87-92%)")
elif test_acc > 70:
    print("✅ GOOD! Model is learning, may need more epochs")
elif test_acc > 55:
    print("⚠️ LEARNING but weak - needs tuning")
else:
    print("❌ STILL NOT LEARNING - fundamental issue")
