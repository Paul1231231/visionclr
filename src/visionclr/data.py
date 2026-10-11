from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets, transforms

IMAGE_SIZE = 128

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
# Referencing UvA DL notebook 
def make_contrastive_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.5,
                                                                     contrast=0.5,
                                                                     saturation=0.5,
                                                                     hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                          transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
                                         ])

class ContrastiveDataset(Dataset):
    def __init__(self, 
                root: str|Path,
                image_sizes: int = IMAGE_SIZE,
    ):
        self.base_dataset = datasets.ImageFolder(root=root, transform=None)
        self.transform = make_contrastive_transform(image_sizes)

    def __len__(self):
            return len(self.base_dataset)

    def __getitem__(self, index: int):
            image, _label = self.base_dataset[index]
            view1 = self.transform(image)
            view2 = self.transform(image)
            return view1, view2

class LabeledImageDataset(Dataset):
    def __init__(self, 
                root: str|Path,
                image_sizes: int = IMAGE_SIZE,
    ):
        self.base_dataset = datasets.ImageFolder(root=root, transform=None)
        self.transform = transforms.Compose([
            transforms.Resize((image_sizes, image_sizes)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
        ])

    def __len__(self):
            return len(self.base_dataset)

    def __getitem__(self, index: int):
            image, label = self.base_dataset[index]
            image = self.transform(image)
            return image, label