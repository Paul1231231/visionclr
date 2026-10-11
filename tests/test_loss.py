
import pytest
import torch

from visionclr.losses import NTXentLoss


def test_loss_is_finite():
    criterion = NTXentLoss(temperature=0.5)

    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)

    loss = criterion(z1, z2)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_gradients_flow():
    criterion = NTXentLoss(temperature=0.5)

    z1 = torch.randn(4, 16, requires_grad=True)
    z2 = torch.randn(4, 16, requires_grad=True)

    loss = criterion(z1, z2)
    loss.backward()

    assert z1.grad is not None
    assert z2.grad is not None
    assert torch.isfinite(z1.grad).all()
    assert torch.isfinite(z2.grad).all()


def test_matching_pairs_have_lower_loss():
    criterion = NTXentLoss(temperature=0.1)

    # Three mutually orthogonal unit vectors.
    # Matching views have cosine similarity 1.
    z1 = torch.eye(3)
    z2 = z1.clone()

    matching_loss = criterion(z1, z2)

    # Permute the second view, so positive partners are mismatched.
    z2_mismatched = z1[[1, 2, 0]]
    mismatched_loss = criterion(z1, z2_mismatched)

    assert matching_loss < mismatched_loss


def test_invalid_temperature():
    with pytest.raises(ValueError):
        NTXentLoss(temperature=0)


def test_batch_size_one_is_rejected():
    criterion = NTXentLoss()

    with pytest.raises(ValueError):
        criterion(torch.randn(1, 8), torch.randn(1, 8))
