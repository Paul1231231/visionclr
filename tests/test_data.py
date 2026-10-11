
import torch
from PIL import Image

from visionclr.data import ContrastiveDataset, make_contrastive_transform


def test_augmentation_output_shape():
    transform = make_contrastive_transform(image_size=128)
    image = Image.new("RGB", (150, 150), color="blue")

    output = transform(image)

    assert output.shape == (3, 128, 128)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_contrastive_dataset_returns_two_views(tmp_path):
    # Create a minimal ImageFolder-compatible dataset.
    for category, color in [
        ("buildings", "red"),
        ("forest", "green"),
    ]:
        folder = tmp_path / category
        folder.mkdir()
        Image.new("RGB", (150, 150), color=color).save(
            folder / "sample.jpg"
        )

    dataset = ContrastiveDataset(tmp_path, image_sizes=128)

    view1, view2 = dataset[0]

    assert len(dataset) == 2
    assert view1.shape == (3, 128, 128)
    assert view2.shape == (3, 128, 128)
    assert view1.dtype == torch.float32
    assert view2.dtype == torch.float32
