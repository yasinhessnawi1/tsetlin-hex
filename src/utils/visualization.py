"""
Visualization utilities for Hex games and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, List, Tuple


def plot_hex_board(
    board: np.ndarray,
    title: str = "Hex Board",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
):
    """
    Plot a Hex board with pieces.

    Args:
        board: Board state array (board_size x board_size)
               0 = empty, 1 = player 0, 2 = player 1
        title: Title for the plot
        figsize: Figure size
        save_path: If provided, save the figure to this path
    """
    board_size = board.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    # Hex cell size
    hex_size = 0.5

    # Colors
    empty_color = '#F0F0F0'
    player0_color = '#FF5555'  # Red
    player1_color = '#5555FF'  # Blue
    border_color = '#333333'

    # Draw hexagons
    for row in range(board_size):
        for col in range(board_size):
            # Calculate center position (offset every other row for hex grid)
            x = col * hex_size * 1.5
            y = row * hex_size * np.sqrt(3) + (col % 2) * hex_size * np.sqrt(3) / 2

            # Determine color based on piece
            cell_value = board[row, col]
            if cell_value == 0:
                color = empty_color
            elif cell_value == 1:
                color = player0_color
            else:
                color = player1_color

            # Draw hexagon
            hexagon = patches.RegularPolygon(
                (x, y), 6, radius=hex_size,
                facecolor=color, edgecolor=border_color, linewidth=1.5
            )
            ax.add_patch(hexagon)

            # Add coordinates
            if cell_value == 0:
                ax.text(x, y, f'{row},{col}', ha='center', va='center',
                       fontsize=6, color='#999999')

    # Set axis limits
    ax.set_xlim(-hex_size, (board_size - 1) * hex_size * 1.5 + hex_size)
    ax.set_ylim(-hex_size, (board_size - 1) * hex_size * np.sqrt(3) + hex_size)

    # Add labels
    ax.text(-hex_size * 2, board_size * hex_size * np.sqrt(3) / 2,
            'Player 0\n(Red)\nTop ↔ Bottom',
            ha='center', va='center', fontsize=10, color=player0_color, weight='bold')

    ax.text((board_size - 1) * hex_size * 1.5 + hex_size * 2, board_size * hex_size * np.sqrt(3) / 2,
            'Player 1\n(Blue)\nLeft ↔ Right',
            ha='center', va='center', fontsize=10, color=player1_color, weight='bold')

    ax.set_title(title, fontsize=14, weight='bold')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Board visualization saved to {save_path}")

    return fig, ax


def plot_training_history(
    history: List[dict],
    save_path: Optional[str] = None
):
    """
    Plot training history (accuracy over epochs).

    Args:
        history: List of dictionaries with 'epoch', 'train_acc', 'test_acc'
        save_path: If provided, save the figure to this path
    """
    epochs = [h['epoch'] for h in history]
    train_accs = [h['train_acc'] for h in history]
    test_accs = [h['test_acc'] for h in history if h['test_acc'] is not None]
    test_epochs = [h['epoch'] for h in history if h['test_acc'] is not None]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax.plot(test_epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2, marker='o')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training Progress', fontsize=14, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add 100% reference line
    ax.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='100%')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    return fig, ax


def plot_confusion_matrix(
    confusion_dict: dict,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.

    Args:
        confusion_dict: Dictionary with 'tn', 'fp', 'fn', 'tp'
        title: Title for the plot
        save_path: If provided, save the figure to this path
    """
    matrix = np.array([
        [confusion_dict['tn'], confusion_dict['fp']],
        [confusion_dict['fn'], confusion_dict['tp']]
    ])

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(matrix, cmap='Blues', aspect='auto')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, matrix[i, j],
                          ha="center", va="center", color="black", fontsize=20, weight='bold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Player 0', 'Player 1'])
    ax.set_yticklabels(['Player 0', 'Player 1'])

    ax.set_xlabel('Predicted', fontsize=12, weight='bold')
    ax.set_ylabel('Actual', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    return fig, ax


def plot_stage_comparison(
    stage_results: dict,
    save_path: Optional[str] = None
):
    """
    Plot comparison of accuracies across different stages.

    Args:
        stage_results: Dictionary mapping stage names to result dictionaries
        save_path: If provided, save the figure to this path
    """
    stages = list(stage_results.keys())
    accuracies = [stage_results[s]['accuracy'] for s in stages]
    p0_accs = [stage_results[s]['player0_accuracy'] for s in stages]
    p1_accs = [stage_results[s]['player1_accuracy'] for s in stages]

    x = np.arange(len(stages))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, accuracies, width, label='Overall', color='#4CAF50')
    bars2 = ax.bar(x, p0_accs, width, label='Player 0', color='#FF5555')
    bars3 = ax.bar(x + width, p1_accs, width, label='Player 1', color='#5555FF')

    ax.set_xlabel('Game Stage', fontsize=12, weight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, weight='bold')
    ax.set_title('Accuracy Comparison Across Game Stages', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=8)

    ax.set_ylim(0, 105)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Stage comparison plot saved to {save_path}")

    return fig, ax


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization utilities...")

    # Create a sample board
    board = np.array([
        [0, 1, 0, 2, 0],
        [1, 0, 2, 0, 1],
        [0, 2, 1, 0, 2],
        [2, 0, 0, 1, 0],
        [0, 1, 2, 0, 1]
    ])

    fig, ax = plot_hex_board(board, title="Sample Hex Game", save_path="test_board.png")
    plt.show()

    print("Visualization test complete!")
