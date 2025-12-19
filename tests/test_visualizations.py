#!/usr/bin/env python3
"""
Test script to verify visualization functions work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.visualization import plot_confusion_matrix, plot_stage_comparison
import matplotlib.pyplot as plt

def test_confusion_matrix():
    """Test confusion matrix plotting."""
    print("Testing confusion matrix visualization...")

    # Sample confusion matrix data
    confusion_dict = {
        'tn': 1450,  # True negatives
        'fp': 50,    # False positives
        'fn': 75,    # False negatives
        'tp': 1425   # True positives
    }

    fig, ax = plot_confusion_matrix(
        confusion_dict,
        title="Test Confusion Matrix - 7x7 End Game",
        save_path="test_confusion_matrix.png"
    )

    plt.close(fig)
    print("✓ Confusion matrix test passed!")


def test_stage_comparison():
    """Test stage comparison plotting."""
    print("Testing stage comparison visualization...")

    # Sample stage results
    stage_results = {
        'end': {
            'accuracy': 95.2,
            'player0_accuracy': 94.8,
            'player1_accuracy': 95.6
        },
        '-2': {
            'accuracy': 87.3,
            'player0_accuracy': 86.1,
            'player1_accuracy': 88.5
        },
        '-5': {
            'accuracy': 72.4,
            'player0_accuracy': 71.8,
            'player1_accuracy': 73.0
        }
    }

    fig, ax = plot_stage_comparison(
        stage_results,
        save_path="test_stage_comparison.png"
    )

    plt.close(fig)
    print("✓ Stage comparison test passed!")


if __name__ == "__main__":
    print("Testing visualization utilities...")
    test_confusion_matrix()
    test_stage_comparison()
    print("All visualization tests passed! ✓")
