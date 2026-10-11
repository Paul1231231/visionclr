
import torch

from visionclr.losses import NTXentLoss


def main():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    batch_size = 32
    embedding_dim = 128

    z1 = torch.randn(
        batch_size, embedding_dim, device=device, requires_grad=True
    )
    z2 = torch.randn(
        batch_size, embedding_dim, device=device, requires_grad=True
    )

    criterion = NTXentLoss(temperature=0.5)
    loss = criterion(z1, z2)

    loss.backward()

    print("Device:", device)
    print("Loss:", loss.item())
    print("Gradient check:", z1.grad is not None and z2.grad is not None)

    assert torch.isfinite(loss)
    assert z1.grad is not None
    assert z2.grad is not None

    print("Contrastive loss check passed!")


if __name__ == "__main__":
    main()
