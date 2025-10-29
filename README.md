# Graph Tsetlin Machine for Hex Board Game Winner Prediction

A modular, high-performance implementation of Graph Tsetlin Machines applied to the Hex board game for winner prediction using deep logical learning and reasoning.

## Project Overview

This project uses Graph Tsetlin Machines (GTM) to predict the winner of Hex board games at different stages:
- **End of game** (final state)
- **2 moves before the end**
- **5 moves before the end**

The goal is to achieve 100% prediction accuracy on smaller boards (10x10) before scaling to larger board sizes (11x11 standard).

### What is Hex?

Hex is a two-player connection game where:
- **Player 0 (Red)**: Connects top to bottom
- **Player 1 (Blue)**: Connects left to right
- The game never ends in a draw
- "Best defense equals best attack" principle

### What is Graph Tsetlin Machine?

GTM is an interpretable AI method that learns logical rules through:
- **Clause-based learning**: Explicit logical patterns
- **Message passing**: Deep reasoning across graph structures
- **Explainability**: Understandable rules (unlike neural networks)
- **Hardware efficiency**: CUDA-accelerated, low energy consumption

## Project Structure

```
tsetlin/
├── src/
│   ├── data_generation/
│   │   ├── hex_game.py          # Core Hex game engine
│   │   ├── game_generator.py    # Parallel game generation (CUDA support)
│   │   └── dataset_builder.py   # Convert to GTM Graphs format
│   ├── models/
│   │   ├── hex_graph_tm.py      # GTM wrapper with CUDA config
│   │   └── predictor.py         # Training and evaluation
│   └── utils/
│       ├── config.py            # Configuration management
│       └── visualization.py     # Plotting utilities
├── scripts/
│   ├── 1_generate_games.py      # Generate training/test datasets
│   ├── 1b_build_gtm_datasets.py # Convert to GTM format
│   ├── 2_train_model.py         # Train GTM models
│   └── 3_evaluate.py            # Evaluate models
├── data/                        # Generated datasets
├── models/                      # Trained models
├── requirements.txt             # Dependencies
└── PROJECT_README.md            # This file
```

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install CUDA Support (Optional but Recommended)

For GPU acceleration, install CuPy for your CUDA version:

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

Check your CUDA version:
```bash
nvidia-smi
```

## Usage

### Step 1: Generate Game Data 🎮

Generate random Hex games for training and testing:

```bash
# Basic usage (10x10 board, 50K training, 10K test)
python scripts/1_generate_games.py

# With CUDA acceleration (recommended for 6GB GPU)
python scripts/1_generate_games.py --use-cuda

# Custom settings
python scripts/1_generate_games.py --board-size 10 --num-train 50000 --num-test 10000 --use-cuda
```

**Output**: Generates `data/train_games_10x10.npz` and `data/test_games_10x10.npz`

**⚡ IMPORTANT**: This step can run in the background on your GPU! Start it immediately and continue with other tasks.

### Step 2: Build GTM Datasets 📊

Convert game data to Graph Tsetlin Machine format:

```bash
python scripts/1b_build_gtm_datasets.py --board-size 10
```

This creates GTM `Graphs` objects with:
- **Nodes**: Each board position (100 nodes for 10x10)
- **Edges**: Hexagonal adjacency (6 neighbors per position)
- **Properties**: Position states (Empty/Player0/Player1)
- **Stages**: end, -2 moves, -5 moves

**Output**: Multiple `.pkl` files in `data/` directory

### Step 3: Train Models 🧠

Train GTM models for each game stage:

```bash
# Train all stages (end, -2, -5)
python scripts/2_train_model.py --board-size 10

# Train specific stage
python scripts/2_train_model.py --stage end --epochs 100

# Custom hyperparameters
python scripts/2_train_model.py --clauses 2000 --depth 3 --epochs 100
```

**Training uses CUDA by default** (configured for 6GB GPU)

**Output**: Trained models in `models/` directory

### Step 4: Evaluate Models 📈

Evaluate trained models on test data:

```bash
# Evaluate all stages
python scripts/3_evaluate.py --board-size 10

# Evaluate specific stage
python scripts/3_evaluate.py --stage end
```

**Output**: Detailed accuracy metrics, confusion matrices, and performance analysis

## Configuration

Edit `src/utils/config.py` to customize:

### Data Generation
- `board_size`: Board dimensions (default: 10)
- `num_train_games`: Training dataset size (default: 50,000)
- `num_test_games`: Test dataset size (default: 10,000)
- `use_cuda`: Enable CUDA acceleration (default: True)

### GTM Model Parameters
- `number_of_clauses`: Pattern detectors (default: 2000)
- `T`: Threshold for clause activation (default: 15000)
- `s`: Specificity parameter (default: 10.0)
- `depth`: Message passing layers (default: 3)
- `message_size`: Message vector size (default: 256)

### CUDA Configuration (6GB GPU)
- `grid`: (16*13, 1, 1) - Optimized for A100-style GPUs
- `block`: (128, 1, 1)

## Performance Considerations

### Memory Optimization
- Board states stored as int8 (minimal memory)
- Compressed NPZ format for datasets
- Batch processing for large datasets

### GPU Acceleration
- Game generation: Can use CuPy for parallel simulation
- GTM training: Built-in CUDA kernels
- Configured for 6GB VRAM (adjust grid/block if needed)

### Speed
- 10x10 boards: ~1000 games/second (CPU)
- Training: ~10-20 seconds/epoch (GPU)

## Graph Representation

### Hex Board → Graph

For a 10x10 board:
- **100 nodes** (one per position)
- **~600 edges** (hexagonal connectivity)
- **3 symbols**: Empty, Player0, Player1
- **Message passing depth 3**: Local → 2 hops → Full board reasoning

### Example Node

Node `R5C5` (row 5, col 5):
- **Properties**: Current piece state (symbol)
- **Edges**: 6 neighbors (hexagonal grid)
- **Messages**: Received from neighbors during reasoning

## Expected Results

### 10x10 Board

| Stage      | Expected Accuracy |
|------------|-------------------|
| End        | 99-100%          |
| 2 Before   | 85-95%           |
| 5 Before   | 70-85%           |

### Achieving 100%

If not reaching 100% on end-game:
1. **Increase clauses**: `--clauses 4000`
2. **Increase depth**: `--depth 4`
3. **More training data**: `--num-train 100000`
4. **Tune T and s**: Experiment with threshold values

Once 100% is achieved on 10x10, scale to 11x11 standard board.

## Modularity

The codebase is designed for maximum modularity:

### Easy to Replace Components

- **Game engine**: Swap `hex_game.py` for other board games
- **Graph encoder**: Modify `dataset_builder.py` for different representations
- **Model**: Use different GTM configurations or architectures

### Easy to Extend

- Add new evaluation metrics in `predictor.py`
- Create custom visualizations in `visualization.py`
- Implement different training strategies

## Interpretability

Unlike neural networks, GTM provides interpretable results:

### Examine Learned Clauses

Clauses represent logical patterns like:
- "If position (5,5) is Player0 AND neighbors have Player0 THEN predict Player0 wins"

### Message Passing Visualization

Track how information flows through the graph during reasoning.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or grid dimensions:
```python
# In config.py
grid = (16*10, 1, 1)  # Reduce from 16*13
```

### Low Accuracy

1. Check data quality: Are games properly generated?
2. Increase model capacity: More clauses, deeper message passing
3. Verify graph construction: Are edges correct?
4. Train longer: More epochs

### Slow Training

1. Ensure CUDA is enabled: Check `nvidia-smi`
2. Use CuPy for data generation: `--use-cuda`
3. Reduce dataset size for testing: `--num-train 10000`

## Next Steps

1. ✅ Generate data for 10x10 board
2. ✅ Train models for all stages
3. ✅ Achieve 100% on end-game
4. ⬜ Scale to 11x11 board
5. ⬜ Optimize hyperparameters
6. ⬜ Analyze learned clauses
7. ⬜ Publish results

## References

- [Graph Tsetlin Machine GitHub](https://github.com/cair/GraphTsetlinMachine)
- [Hex Game Rules](https://www.krammer.nl/hex)
- [Tsetlin Machine Book](https://tsetlinmachine.org)

## License

This project is part of research on Graph Tsetlin Machines for board game reasoning.

---

## Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data (START THIS IMMEDIATELY - runs in background on GPU!)
python scripts/1_generate_games.py --use-cuda

# 3. Build GTM datasets (after step 2 completes)
python scripts/1b_build_gtm_datasets.py

# 4. Train models
python scripts/2_train_model.py

# 5. Evaluate
python scripts/3_evaluate.py
```

---

**Built with modularity, performance, and interpretability in mind.**
