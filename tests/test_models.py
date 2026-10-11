
import torch

from visionclr.models import SimCLRModel


def test_simclr_output_shapes():
    model = SimCLRModel(
        projection_dim=128,
        hidden_dim=512,
    )

    x = torch.randn(4, 3, 128, 128)
    h, z = model(x)

    assert h.shape == (4, 512)
    assert z.shape == (4, 128)


def test_simclr_gradients():
    model = SimCLRModel()
    x = torch.randn(2, 3, 128, 128)

    h, z = model(x)
    loss = z.square().mean()
    loss.backward()

    assert model.encoder[0].weight.grad is not None
    assert model.projector[0].weight.grad is not None
