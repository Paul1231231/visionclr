
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Make src/visionclr importable when running this script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionclr.data import ContrastiveDataset


def find_image_root(split: str) -> Path:
    """Find the directory immediately above the class folders."""
    candidates = [
        Path("data/raw") / split / split,
        Path("data/raw") / split,
    ]

    for candidate in candidates:
        if candidate.is_dir() and any(
            (child / "buildings").is_dir()
            for child in candidate.iterdir()
            if child.is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        f"Could not find the class folders for {split}. "
        "Check the extracted Kaggle dataset layout."
    )


def main():
    train_root = find_image_root("seg_train")
    print(f"Training image directory: {train_root}")

    dataset = ContrastiveDataset(train_root, image_size=128)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    view1, view2 = next(iter(loader))

    print(f"Number of training images: {len(dataset)}")
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")
    print(f"View 1 dtype: {view1.dtype}")
    print(f"View 1 device: {view1.device}")

    # Undo normalization for visualization.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))


    for i in range(4):
        for col, batch, view_name in [
            (0, view1, "A"),
            (1, view2, "B"),
        ]:
            image = (batch[i].cpu() * std + mean).clamp(0, 1)
            image = image.permute(1, 2, 0).numpy()

            axes[i, col].imshow(image)
            axes[i, col].set_title(f"Image {i + 1}, view {view_name}")

    # Hide unused columns in this first visualization.
    for ax in axes.flat:
        ax.axis("off")


    plt.tight_layout()
    output_path = Path("outputs/augmentations.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
