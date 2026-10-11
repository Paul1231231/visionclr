
import torch

from visionclr.models import SimCLRModel


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = SimCLRModel(
        projection_dim=128,
        hidden_dim=512,
    ).to(device)

    # Simulate one batch from Stage 1.
    x = torch.randn(32, 3, 128, 128, device=device)

    h, z = model(x)

    print("Device:", device)
    print("Input shape:", x.shape)
    print("Representation shape:", h.shape)
    print("Projection shape:", z.shape)

    total_params = sum(
        p.numel() for p in model.parameters()
    )
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    assert h.shape == (32, 512)
    assert z.shape == (32, 128)
    assert torch.isfinite(h).all()
    assert torch.isfinite(z).all()

    print("Model forward pass passed!")


if __name__ == "__main__":
    main()
