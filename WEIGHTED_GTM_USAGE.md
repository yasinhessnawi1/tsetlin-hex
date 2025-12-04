# Weighted GTM Usage Guide

## Purpose

The Weighted GTM addresses the **class imbalance problem** you saw in your 5x5 training results:
- Player 0: 83.46% accuracy
- Player 1: 62.06% accuracy
- **21% gap** between classes

This happens because your training data has 57.6% Player 0 wins and only 42.4% Player 1 wins.

## How It Works

The WeightedGTM uses **class balancing through oversampling**:

1. Identifies minority class (Player 1)
2. Replicates minority samples to match majority class count
3. Trains on balanced dataset (50% Player 0, 50% Player 1)
4. Model learns both classes equally well

## Quick Start

### Option 1: Use the Training Script

```bash
run_weighted_gtm.bat
```

This will:
- Load 10K training samples and 2K test samples
- Balance the classes automatically
- Train for 200 epochs with best parameters from Phase 1
- Show per-class accuracy to verify the gap is reduced

### Option 2: Use in Python Code

```python
from src.models.weighted_gtm import WeightedGTM
from src.data.data_loader import load_processed_data

# Load your data
train_graphs, train_labels = load_processed_data("path/to/train.npz")
test_graphs, test_labels = load_processed_data("path/to/test.npz")

# Create weighted GTM with class balancing
model = WeightedGTM(
    number_of_clauses=200,
    T=10000,
    s=10.0,
    depth=3,
    class_weight='balanced'  # Enable class balancing
)

# Train (balancing happens automatically)
model.fit(train_graphs, train_labels, epochs=200)

# Predict
predictions = model.predict(test_graphs)

# Evaluate
accuracy = model.evaluate(test_graphs, test_labels)
print(f"Accuracy: {accuracy:.2f}%")
```

## Parameters

### Standard GTM Parameters
- `number_of_clauses`: Number of clauses (default: 200)
- `T`: Threshold (default: 10000 from your Phase 1 results)
- `s`: Specificity (default: 10.0 from your Phase 1 results)
- `depth`: Message passing depth (default: 3 from your Phase 1 results)

### Class Balancing Parameter
- `class_weight`:
  - `'balanced'` - Automatically balance classes (recommended for your data)
  - `'none'` - No balancing (standard GTM behavior)

## Expected Results

**Before (Standard GTM):**
- Overall: 74.55%
- Player 0: 83.46%
- Player 1: 62.06%
- Gap: 21.4%

**After (Weighted GTM with balancing):**
- Overall: Should be similar or better (~74-78%)
- Player 0: Slightly lower (~75-80%)
- Player 1: **Significantly improved** (~70-75%)
- Gap: **Much smaller** (~5-10%)

The goal is not necessarily higher overall accuracy, but **balanced performance** across both classes.

## When to Use

✅ **Use Weighted GTM when:**
- You have class imbalance (like your 57/43 split)
- Per-class accuracy gap is large (>15%)
- You need balanced predictions for both classes

❌ **Don't use Weighted GTM when:**
- Classes are already balanced (50/50 split)
- You specifically want to optimize for one class
- Memory is extremely limited (balancing increases training samples)

## Implementation Details

The implementation:
1. **Analyzes** training data class distribution
2. **Replicates** minority class samples to match majority count
3. **Creates** new balanced Graphs object with duplicated samples
4. **Shuffles** to ensure random order
5. **Trains** base GTM on balanced dataset

This is a simple but effective approach that works with the existing GTM API without requiring internal modifications.

## Comparison with Baseline

Your 5x5 200-epoch training showed:
```
Final Training Accuracy: 74.65%
Final Test Accuracy: 74.55%

Per-class (Test):
  Player 0: 83.46%
  Player 1: 62.06%
  Gap: 21.40%
```

Run `run_weighted_gtm.bat` to see if class balancing reduces this gap!
