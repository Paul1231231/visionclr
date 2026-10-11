
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

from visionclr.data import LabeledImageDataset
from visionclr.models import SimCLRModel


IMAGE_SIZE = 128
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_model(checkpoint_path, device):
    model = SimCLRModel()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, loader, device):
    all_features = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)

        # Use the encoder's 512-dimensional features, not the projector.
        features = model.encoder(images)
        features = F.normalize(features, p=2, dim=1)

        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


@torch.no_grad()
def extract_image_feature(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image = make_transform()(image).unsqueeze(0).to(device)

    feature = model.encoder(image)
    return F.normalize(feature, p=2, dim=1).cpu()


def retrieve(query_feature, gallery_features, gallery_dataset, top_k=5):
    similarities = query_feature @ gallery_features.T
    k = min(top_k, len(gallery_dataset))
    scores, indices = similarities[0].topk(k)

    print(f"\nTop {k} most similar training images:")
    for rank, (score, index) in enumerate(zip(scores, indices), start=1):
        image_path, label = gallery_dataset.base_dataset.samples[index.item()]
        class_name = gallery_dataset.base_dataset.classes[label]

        print(
            f"{rank}. {image_path}\n"
            f"   Class: {class_name} | "
            f"Cosine similarity: {score.item():.4f}"
        )


def evaluate_test_set(
    test_features,
    test_labels,
    gallery_features,
    gallery_labels,
    class_names,
):
    # Each test image is classified by its closest training image.
    similarities = test_features @ gallery_features.T
    nearest_indices = similarities.argmax(dim=1)
    predictions = gallery_labels[nearest_indices]

    accuracy = (predictions == test_labels).float().mean().item()
    print(f"\n1-nearest-neighbor test accuracy: {accuracy:.2%}")

    print("\nPer-class accuracy:")
    for class_index, class_name in enumerate(class_names):
        mask = test_labels == class_index
        if mask.any():
            class_accuracy = (
                predictions[mask] == test_labels[mask]
            ).float().mean().item()
            print(f"{class_name}: {class_accuracy:.2%}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="SimCLR image retrieval and 1-NN test evaluation"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/simclr/simclr_checkpoint.pt"),
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("data/raw/seg_train/seg_train"),
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("data/raw/seg_test/seg_test"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional custom image or a single image from the test set",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    if args.top_k < 1 or args.batch_size < 1:
        raise ValueError("top-k and batch-size must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.train_root.is_dir():
        raise FileNotFoundError(f"Training directory not found: {args.train_root}")
    if args.evaluate_test and not args.test_root.is_dir():
        raise FileNotFoundError(f"Test directory not found: {args.test_root}")
    if args.image is not None and not args.image.is_file():
        raise FileNotFoundError(f"Query image not found: {args.image}")
    if args.image is None and not args.evaluate_test:
        parser.error("Provide --image, --evaluate-test, or both.")

    device = choose_device()
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    train_dataset = LabeledImageDataset(args.train_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Extracting features from {len(train_dataset)} training images...")
    train_features, train_labels = extract_features(
        model, train_loader, device
    )
    print(f"Gallery feature matrix: {tuple(train_features.shape)}")

    if args.image is not None:
        query_feature = extract_image_feature(model, args.image, device)
        print(f"\nQuery image: {args.image}")
        retrieve(
            query_feature,
            train_features,
            train_dataset,
            top_k=args.top_k,
        )

    if args.evaluate_test:
        test_dataset = LabeledImageDataset(args.test_root)

        if test_dataset.base_dataset.classes != train_dataset.base_dataset.classes:
            raise ValueError("Train and test class mappings do not match.")

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        print(f"\nExtracting features from {len(test_dataset)} test images...")
        test_features, test_labels = extract_features(
            model, test_loader, device
        )

        evaluate_test_set(
            test_features,
            test_labels,
            train_features,
            train_labels,
            train_dataset.base_dataset.classes,
        )


if __name__ == "__main__":
    main()
