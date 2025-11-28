"""
Trace the complete data flow from raw data to GTM prediction.
Verify that graphs and labels are passed correctly at each step.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pickle
from src.models import HexGraphTM, Predictor

print("="*70)
print("TRACING DATA FLOW: Raw Data -> GTM -> Predictions")
print("="*70)

# Step 1: Load GTM graphs
print("\n" + "="*70)
print("STEP 1: Load GTM Graphs")
print("="*70)

with open('data/train_gtm_5x5_end.pkl', 'rb') as f:
    data = pickle.load(f)

graphs = data['graphs']
labels = data['labels']

print(f"Loaded graphs type: {type(graphs)}")
print(f"Graphs.number_of_graphs: {graphs.number_of_graphs}")
print(f"Labels shape: {labels.shape}")
print(f"Labels dtype: {labels.dtype}")
print(f"First 5 labels: {labels[:5]}")

# Step 2: Create GTM model
print("\n" + "="*70)
print("STEP 2: Create GTM Model")
print("="*70)

model = HexGraphTM(
    number_of_clauses=100,  # Small for quick test
    T=100,
    s=10.0,
    depth=2,
    message_size=256,
    message_bits=2,
    max_included_literals=32,
    grid=(16*8, 1, 1),
    block=(128, 1, 1)
)

print(f"Model type: {type(model)}")
print(f"Model.tm before fit: {model.tm}")

# Step 3: Fit the model (1 epoch only for testing)
print("\n" + "="*70)
print("STEP 3: Fit Model (Testing 1 epoch)")
print("="*70)

print("Calling model.fit()...")
model.fit(graphs, labels, epochs=1)

print(f"Model.tm after fit: {type(model.tm)}")
print(f"Model trained: {model.trained}")

# Step 4: Make predictions
print("\n" + "="*70)
print("STEP 4: Make Predictions")
print("="*70)

print("Calling model.predict()...")
predictions = model.predict(graphs)

print(f"Predictions type: {type(predictions)}")
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions dtype: {predictions.dtype}")
print(f"First 10 predictions: {predictions[:10]}")
print(f"First 10 labels:      {labels[:10]}")

# Step 5: Check prediction distribution
print("\n" + "="*70)
print("STEP 5: Check Prediction Distribution")
print("="*70)

unique_preds, counts_preds = np.unique(predictions, return_counts=True)
unique_labels, counts_labels = np.unique(labels, return_counts=True)

print("Predictions distribution:")
for val, count in zip(unique_preds, counts_preds):
    print(f"  Class {val}: {count} ({100*count/len(predictions):.1f}%)")

print("\nTrue labels distribution:")
for val, count in zip(unique_labels, counts_labels):
    print(f"  Class {val}: {count} ({100*count/len(labels):.1f}%)")

# Step 6: Check if predictions match shape
print("\n" + "="*70)
print("STEP 6: Verify Prediction/Label Alignment")
print("="*70)

if predictions.shape == labels.shape:
    print("[OK] Predictions and labels have same shape")
else:
    print(f"[ERROR] Shape mismatch! Predictions: {predictions.shape}, Labels: {labels.shape}")

if len(predictions) == graphs.number_of_graphs:
    print("[OK] Number of predictions matches number of graphs")
else:
    print(f"[ERROR] Count mismatch! Predictions: {len(predictions)}, Graphs: {graphs.number_of_graphs}")

# Step 7: Calculate accuracy
print("\n" + "="*70)
print("STEP 7: Calculate Accuracy")
print("="*70)

accuracy = np.sum(predictions == labels) / len(labels) * 100
print(f"Accuracy: {accuracy:.2f}%")

if accuracy > 50:
    print("[INFO] Accuracy > 50% - model is learning something")
else:
    print("[INFO] Accuracy <= 50% - model not learning or predicting opposite")

# Step 8: Check if model is just predicting majority class
print("\n" + "="*70)
print("STEP 8: Check for Majority Class Prediction")
print("="*70)

majority_class = 0 if np.sum(labels == 0) > np.sum(labels == 1) else 1
majority_pct = np.sum(labels == majority_class) / len(labels) * 100

print(f"Majority class: {majority_class}")
print(f"Majority percentage: {majority_pct:.2f}%")
print(f"Model accuracy: {accuracy:.2f}%")

if abs(accuracy - majority_pct) < 1.0:
    print("\n[WARNING] Model is likely just predicting majority class!")
    print("This means it's not learning the actual patterns.")
else:
    print("\n[OK] Model is not just predicting majority class")

# Final summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Graphs object loaded correctly: {graphs.number_of_graphs} graphs")
print(f"✓ Labels loaded correctly: {len(labels)} labels")
print(f"✓ Model fit completed without errors")
print(f"✓ Predictions generated: {len(predictions)} predictions")
print(f"  Model accuracy: {accuracy:.2f}%")
print(f"  Baseline (majority): {majority_pct:.2f}%")

if abs(accuracy - majority_pct) < 1.0:
    print("\n[ISSUE] Model is stuck at baseline - not learning patterns")
    print("Possible causes:")
    print("  1. Task is too difficult for current parameters")
    print("  2. Graph structure doesn't capture necessary information")
    print("  3. Need more epochs / stronger parameters")
    print("  4. Feature representation is insufficient")
else:
    print("\n[SUCCESS] Model is learning beyond baseline!")

print("="*70)

