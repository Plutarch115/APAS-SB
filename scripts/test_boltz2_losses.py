"""
Test script for Boltz-2 loss functions.

Tests:
1. Huber Loss
2. Pairwise Ranking Loss
3. Focal Loss
4. Combined Loss
"""

import sys
sys.path.append('.')

import torch
from pearl.training.boltz2_losses import (
    HuberLoss,
    PairwiseRankingLoss,
    FocalLoss,
    CombinedBoltz2Loss,
    get_loss_function
)


def test_huber_loss():
    """Test Huber Loss"""
    print("\n" + "=" * 60)
    print("Testing Huber Loss")
    print("=" * 60)

    loss_fn = HuberLoss(delta=1.0)

    # Test case 1: Small errors (should use L2)
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.1, 2.1, 3.1])
    loss = loss_fn(preds, targets)
    print(f"✓ Small errors (L2 regime): {loss.item():.4f}")

    # Test case 2: Large errors (should use L1)
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([5.0, 8.0, 10.0])
    loss = loss_fn(preds, targets)
    print(f"✓ Large errors (L1 regime): {loss.item():.4f}")

    # Test case 3: With weights
    weights = torch.tensor([1.0, 2.0, 3.0])
    loss = loss_fn(preds, targets, weights)
    print(f"✓ Weighted loss: {loss.item():.4f}")


def test_pairwise_ranking_loss():
    """Test Pairwise Ranking Loss"""
    print("\n" + "=" * 60)
    print("Testing Pairwise Ranking Loss")
    print("=" * 60)

    loss_fn = PairwiseRankingLoss(margin=0.5)

    # Test case 1: Correct ranking
    preds = torch.tensor([3.0, 2.0, 1.0])
    targets = torch.tensor([3.0, 2.0, 1.0])
    loss = loss_fn(preds, targets)
    print(f"✓ Correct ranking: {loss.item():.4f} (should be ~0)")

    # Test case 2: Incorrect ranking
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([3.0, 2.0, 1.0])
    loss = loss_fn(preds, targets)
    print(f"✓ Incorrect ranking: {loss.item():.4f} (should be >0)")

    # Test case 3: Partial ranking
    preds = torch.tensor([2.5, 2.0, 1.0])
    targets = torch.tensor([3.0, 2.0, 1.0])
    loss = loss_fn(preds, targets)
    print(f"✓ Partial ranking: {loss.item():.4f}")


def test_focal_loss():
    """Test Focal Loss"""
    print("\n" + "=" * 60)
    print("Testing Focal Loss")
    print("=" * 60)

    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # Test case 1: Easy positives (high confidence, correct)
    preds = torch.tensor([0.9, 0.95, 0.99])
    targets = torch.tensor([1.0, 1.0, 1.0])
    loss = loss_fn(preds, targets)
    print(f"✓ Easy positives: {loss.item():.4f} (should be small)")

    # Test case 2: Hard positives (low confidence, correct)
    preds = torch.tensor([0.6, 0.55, 0.51])
    targets = torch.tensor([1.0, 1.0, 1.0])
    loss = loss_fn(preds, targets)
    print(f"✓ Hard positives: {loss.item():.4f} (should be larger)")

    # Test case 3: Easy negatives
    preds = torch.tensor([0.1, 0.05, 0.01])
    targets = torch.tensor([0.0, 0.0, 0.0])
    loss = loss_fn(preds, targets)
    print(f"✓ Easy negatives: {loss.item():.4f} (should be small)")

    # Test case 4: Hard negatives
    preds = torch.tensor([0.4, 0.45, 0.49])
    targets = torch.tensor([0.0, 0.0, 0.0])
    loss = loss_fn(preds, targets)
    print(f"✓ Hard negatives: {loss.item():.4f} (should be larger)")


def test_combined_loss():
    """Test Combined Boltz-2 Loss"""
    print("\n" + "=" * 60)
    print("Testing Combined Boltz-2 Loss")
    print("=" * 60)

    loss_fn = CombinedBoltz2Loss()

    # Test case 1: Mixed batch (continuous + binary)
    preds = torch.tensor([2.5, 3.0, 0.8, 0.2, 1.5])
    targets = torch.tensor([2.0, 3.5, 1.0, 0.0, 2.0])
    task_types = ['binding_affinity', 'binding_affinity', 'binary_classification', 
                  'binary_classification', 'binding_affinity']
    weights = torch.ones(5)

    total_loss, loss_dict = loss_fn(preds, targets, task_types, weights)
    print(f"✓ Mixed batch loss: {total_loss.item():.4f}")
    print(f"  Components: {loss_dict}")

    # Test case 2: Only continuous
    preds = torch.tensor([2.5, 3.0, 1.5])
    targets = torch.tensor([2.0, 3.5, 2.0])
    task_types = ['binding_affinity', 'binding_affinity', 'binding_affinity']
    weights = torch.ones(3)

    total_loss, loss_dict = loss_fn(preds, targets, task_types, weights)
    print(f"\n✓ Continuous only: {total_loss.item():.4f}")
    print(f"  Components: {loss_dict}")

    # Test case 3: Only binary
    preds = torch.tensor([0.8, 0.2, 0.9])
    targets = torch.tensor([1.0, 0.0, 1.0])
    task_types = ['binary_classification', 'binary_classification', 'binary_classification']
    weights = torch.ones(3)

    total_loss, loss_dict = loss_fn(preds, targets, task_types, weights)
    print(f"\n✓ Binary only: {total_loss.item():.4f}")
    print(f"  Components: {loss_dict}")


def test_get_loss_function():
    """Test loss function factory"""
    print("\n" + "=" * 60)
    print("Testing Loss Function Factory")
    print("=" * 60)

    task_types = ['binding_affinity', 'binary_classification', 'ddg', 'kcat', 'fitness']

    for task_type in task_types:
        loss_fn = get_loss_function(task_type)
        print(f"✓ {task_type}: {loss_fn.__class__.__name__}")


if __name__ == '__main__':
    print("\n" + "🧪" * 40)
    print("BOLTZ-2 LOSS FUNCTIONS TEST SUITE")
    print("🧪" * 40)

    test_huber_loss()
    test_pairwise_ranking_loss()
    test_focal_loss()
    test_combined_loss()
    test_get_loss_function()

    print("\n" + "=" * 60)
    print("✅ ALL LOSS FUNCTION TESTS COMPLETED!")
    print("=" * 60)

