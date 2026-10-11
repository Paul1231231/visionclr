
import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from visionclr.data import ContrastiveDataset
from visionclr.losses import NTXentLoss
from visionclr.models import SimCLRModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_loss_plot(losses: list[float], path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent Loss")
    plt.title("SimCLR Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/seg_train/seg_train"),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Maximum batches per epoch; 0 means all batches")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/simclr"))
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError("epochs must be >= 1 and batch-size must be >= 2")
    if args.max_batches < 0 or args.num_workers < 0:
        raise ValueError("max-batches and num-workers cannot be negative")

    set_seed(args.seed)

    if not args.data_root.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {args.data_root}\n"
            "Check the extracted dataset path and pass --data-root."
        )

    dataset = ContrastiveDataset(args.data_root)
    if len(dataset) < 2:
        raise ValueError("Need at least two training images.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=False,
    )

    if len(loader) == 0:
        raise ValueError(
            "No complete batches. Reduce --batch-size or add more images."
        )

    device = choose_device()
    model = SimCLRModel().to(device)
    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "simclr_checkpoint.pt"
    log_path = args.output_dir / "training_log.csv"
    plot_path = args.output_dir / "loss_curve.png"

    print(f"Device: {device}")
    print(f"Training images: {len(dataset)}")
    print(f"Classes found: {dataset.base_dataset.classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")

    epoch_losses: list[float] = []

    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mean_loss", "batches"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            batches_seen = 0

            for batch_idx, (view1, view2) in enumerate(loader):
                if args.max_batches and batch_idx >= args.max_batches:
                    break

                view1 = view1.to(device)
                view2 = view2.to(device)

                optimizer.zero_grad(set_to_none=True)

                _, z1 = model(view1)
                _, z2 = model(view2)
                loss = criterion(z1, z2)

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at epoch {epoch}, "
                        f"batch {batch_idx + 1}: {loss.item()}"
                    )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batches_seen += 1

                if (batch_idx + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch}/{args.epochs} | "
                        f"Batch {batch_idx + 1}/{len(loader)} | "
                        f"Loss: {loss.item():.4f}"
                    )

            if batches_seen == 0:
                raise RuntimeError("No batches were processed.")

            mean_loss = total_loss / batches_seen
            epoch_losses.append(mean_loss)
            writer.writerow([epoch, f"{mean_loss:.6f}", batches_seen])
            f.flush()

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": mean_loss,
                    "model_config": {
                        "projection_dim": 128,
                        "hidden_dim": 512,
                    },
                    "image_size": 128,
                    "temperature": args.temperature,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

            save_loss_plot(epoch_losses, plot_path)
            print(
                f"Epoch {epoch}/{args.epochs} complete | "
                f"Mean loss: {mean_loss:.4f}"
            )

    print("\nTraining finished.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training log: {log_path}")
    print(f"Loss curve: {plot_path}")


if __name__ == "__main__":
    main()
