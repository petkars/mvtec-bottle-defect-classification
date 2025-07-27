import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
import torch

# CONFIRM: Using CPU only
device = torch.device("cpu")
print(f"Running on device: {device}")

# Paths
DATA_DIR = Path("data/bottle")
TRAIN_DIR = DATA_DIR / "train" / "good"
TEST_DIR = DATA_DIR / "test"

# -------------------------
# Normalization Explained
# -------------------------
# Pixel values in an image range from 0 to 255 (for 8-bit images).
# Neural networks learn better when input data has small, consistent values.
# So we normalize: divide by 255, and often subtract the mean & divide by std dev (here using ImageNet stats).
# This makes training stable and faster.

# Define transformation: resize, convert to tensor, normalize
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize for model input
    transforms.ToTensor(),  # Converts [0,255] to [0,1]
    transforms.Normalize([0.485, 0.456, 0.406],  # mean (R,G,B)
                         [0.229, 0.224, 0.225])  # std (R,G,B)
])

# -------------------------
# Custom Dataset
# -------------------------
class BottleDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.images = []
        self.labels = []

        if mode == 'train':
            # All train images are "good"
            for img_file in sorted((self.root_dir).glob("*.png")):
                self.images.append(img_file)
                self.labels.append("good")
        else:
            # Test images may be good or defective
            for defect_type_dir in sorted((self.root_dir).iterdir()):
                for img_file in defect_type_dir.glob("*.png"):
                    self.images.append(img_file)
                    self.labels.append(defect_type_dir.name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Load datasets
train_dataset = BottleDataset(root_dir=TRAIN_DIR, transform=transform, mode='train')
test_dataset = BottleDataset(root_dir=TEST_DIR, transform=transform, mode='test')

print(f"✅ Loaded {len(train_dataset)} training images (all 'good')")
print(f"✅ Loaded {len(test_dataset)} test images with labels: {set(test_dataset.labels)}")

# -------------------------
# Visualize samples
# -------------------------
def show_samples(dataset, title="Samples", num=6):
    plt.figure(figsize=(15, 5))
    for i in range(num):
        img, label = dataset[i]
        img = img.permute(1, 2, 0)  # Convert from [C,H,W] to [H,W,C] for matplotlib
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Denormalize
        img = img.clamp(0, 1)
        plt.subplot(1, num, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
    plt.suptitle(title)
    plt.show()


show_samples(train_dataset, title="Train - Good Samples")
show_samples(test_dataset, title="Test - Mixed Defect Types")

