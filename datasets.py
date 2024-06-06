import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import albumentations as A

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


########## Utils ###########################################################
def show_images(images, num_rows=1, mapping=None, title=None):
    """
    Display a set of images in a grid.
    
    Parameters:
        images (list or ndarray): torch tensor on dim (2, n, 1, 28, 28)
        num_rows (int): Number of rows in the image grid.
        title (str): Optional title for the overall figure.
    """
    num_images = len(images[0])
    num_cols = np.ceil(num_images / num_rows)  # Calculate number of columns

    # Create a figure with subplots in a grid
    fig, axes = plt.subplots(num_rows, int(num_cols), figsize=(2 * num_cols, 2 * num_rows))
    
    if title:
        plt.suptitle(title)  # Set the title for the entire figure

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            # Display image
            ax.imshow(images[0][i][0].numpy(), cmap='gray')
            ax.set_title(mapping[images[1][i].item()])
            ax.axis('off')  # Hide axes
        else:
            ax.axis('off')  # Hide axes for empty subplots

    plt.tight_layout()
    plt.show()


def print_tensor_stats(tensor, name):
    print(f"{name}: min={tensor.min().item()}, max={tensor.max().item()}, mean={tensor.mean().item()}, std={tensor.std().item()}")





########## EMNIST DATASET ###########################################################
class EmnistDataset(Dataset):
    def __init__(self, path, transform=None, limit=None):
        self.path = path
        self.transform = transform 
        dataset = pd.read_csv(path, header=None)
        if limit:
            dataset = dataset.sample(n=limit)
        X, y = dataset[list(range(1, dataset.shape[1]))], dataset[0]
        self.X = X.to_numpy(dtype='float32').reshape((-1, 28, 28))
        self.y = y.to_numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slicing
            images = np.expand_dims(self.X[idx], axis=3) / 255.0
            if self.transform:
                images = torch.stack([self.transform(img) for img in images])  # Apply transform per image
            labels = torch.tensor(self.y[idx], dtype=torch.long)
            return images, labels
        else:
            image = np.expand_dims(self.X[idx], axis=2) / 255.0
            if self.transform:
                image = self.transform(image)
            label = torch.tensor(self.y[idx], dtype=torch.long)
            return image, label

class EmnistMapping:
    def __init__(self, path):
        with open(path, "r") as f:
            lines = f.read().split("\n")[:-1]
            self.mapping = {}
            self.unmapping = {}
            for line in lines:
                char_class, char_code = line.split()
                self.mapping[int(char_class)] = chr(int(char_code))
                self.unmapping[chr(int(char_code))] = int(char_class)

    def __getitem__(self, key):
        return self.mapping[key]

    def __str__(self):
        return str(self.mapping)

emnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation((-90, -90)),          # Rotate by exactly 90 degrees
        transforms.RandomHorizontalFlip(p=1),
        transforms.Normalize((0.5,), (0.5,))  # Normalizes tensor to have mean around 0 and variance around 1
    ])





########## EHC DATASET ###########################################################
class EhcDataset(Dataset):
    def __init__(self, path, transform=None, limit=None):
        self.path = path
        self.transform = transform 
        self.dataset = pd.read_csv(f"{path}/english.csv")
        self.mapping = lambda label: ord(label) - 48
        self.unmapping = lambda label: chr(label + 48)
        if limit:
            self.dataset = self.dataset.sample(n=limit)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            images_path = self.dataset.iloc[idx]["image"]
            images = [Image.open(f"{self.path}/{path}") for path in images_path]
            if self.transform:
                images = torch.stack([self.transform(img) for img in images])  # Apply transform per image
                images = images.unsqueeze(1)  # Add channel dimension: [Batch, 1, 28, 28]

            labels = list(map(self.mapping, self.dataset.iloc[idx]["label"]))
            labels = torch.tensor(labels, dtype=torch.long)
            return images, labels

        else:
            image_path = self.dataset.iloc[idx]["image"]
            image = Image.open(f"{self.path}/{image_path}")
            if self.transform:
                image = self.transform(image)
                # image = image.unsqueeze(0) 
                # print_tensor_stats(image, "Transformed Image")

            label = self.mapping(self.dataset.iloc[idx]["label"])
            label = torch.tensor(label, dtype=torch.long)
            return image, label

ehc_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(1),
        transforms.Lambda(lambda image: 1 - image),  # Invert the image
        transforms.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize to [-1, 1]
        transforms.CenterCrop((900, 900)),
        transforms.Resize((28, 28))
    ])




########## CHOICE DATASET ###########################################################
class ChoicetDataset(Dataset):
    def __init__(self, path, transform=None, limit=None):
        self.path = path
        self.transform = transform
        self.dataset = pd.DataFrame(columns=["path", "label"])
        for dir in os.listdir(path + "/data"):
            new_rows = {"path": os.listdir(path + "/data/" + dir), "label": len(os.listdir(path + "/data/" + dir)) * [dir]}
            new_dataset = pd.DataFrame(new_rows)
            self.dataset = pd.concat([self.dataset, new_dataset], ignore_index=True)
        if limit:
            self.dataset = self.dataset.sample(limit)
        with open(path + "/label.txt", "r") as f:
            lines = f.read().split("\n")
            self.mapping = {idx: char for idx, char in enumerate(lines)}
            self.unmapping = {char: idx for idx, char in enumerate(lines)}

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            images_path = list(self.dataset.iloc[idx]['path'])
            labels = list(self.dataset.iloc[idx]['label'].astype(int))
            images = [Image.open(f"{self.path}/data/{labels[i]}/{images_path[i]}") for i in range(len(labels))]
            if self.transform:
                images = torch.stack([self.transform(image) for image in images])
                # images = images.unsqueeze(1)
            labels = torch.tensor(labels, dtype=torch.long)
            return images, labels
        else:
            image_path = self.dataset.iloc[idx]['path']
            label = self.dataset.iloc[idx]['label']
            image = Image.open(f"{self.path}/data/{label}/{image_path}")
            if self.transform:
                image = self.transform(image)
            label = torch.tensor(int(label), dtype=torch.long)
            return image, label

choice_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda image: 1 - image),  # Invert the image
        transforms.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize to [-1, 1]
    ])




########## FONT DATASET ###########################################################
class FontDataset:
    def __init__(self, dataset_path, mapping_path, transform=None, limit=None):
        self.path = dataset_path
        self.dataset = pd.read_csv(self.path)
        self.transform = transform
        with open(mapping_path, "r") as f:
            lines = f.read().split("\n")[:-1]
            self.mapping = {}
            self.unmapping = {}
            for line in lines:
                char_class, char_code = line.split()
                self.mapping[int(char_class)] = chr(int(char_code))
                self.unmapping[chr(int(char_code))] = int(char_class)
        if limit:
            self.dataset = self.dataset.sample(limit)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            images_path = self.dataset.iloc[idx]["path"]
            images = [Image.open(path) for path in images_path]
            if self.transform:
                images = torch.stack([self.transform(np.array(img)) for img in images])  # Apply transform per image
                # images = images.unsqueeze(1)  # Add channel dimension: [Batch, 1, 28, 28]

            labels = [self.unmapping[label] for label in self.dataset.iloc[idx]["label"]]
            labels = torch.tensor(labels, dtype=torch.long)
            return images, labels

        else:
            image_path = self.dataset.iloc[idx]["path"]
            image = Image.open(image_path)
            if self.transform:
                image = np.array(image)
                image = self.transform(image)
                # image = image.unsqueeze(0) 
                # print_tensor_stats(image, "Transformed Image")

            label = self.unmapping[self.dataset.iloc[idx]["label"]]
            label = torch.tensor(label, dtype=torch.long)
            return image, label


font_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(1),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize to [-1, 1]
    ])




########## AUGMENATIONS ###########################################################
augmentation_transforms = A.Compose([
    A.Rotate(limit=20, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.3),
    A.ElasticTransform(alpha=0.3, sigma=50, alpha_affine=3, p=0.2),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.GaussNoise(var_limit=(0, 0.05), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.CoarseDropout(max_holes=2, max_height=6, max_width=6, p=0.1, fill_value=0)
])

# Определение функции для применения трансформаций Albumentations
def augmentation_transform(img):
    return augmentation_transforms(image=img)['image']

# Определение трансформаций torchvision
augmentations = transforms.Compose([
    transforms.Lambda(augmentation_transform),
    transforms.ToTensor(),
    transforms.RandomRotation((-90, -90)),          
    transforms.RandomHorizontalFlip(p=1),
    transforms.Normalize((0.5,), (0.5,))
])

augmented_font_transform = transforms.Compose([
        transforms.Lambda(augmentation_transform),
        transforms.ToTensor(),
        transforms.Grayscale(1),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize to [-1, 1]
    ])

########## GAN DATASET ###########################################################
class GanDataset:
    def __init__(self, dataset_path, mapping_path, transform=None, limit=None):
        self.path = dataset_path
        self.dataset = pd.read_csv(self.path)
        self.transform = transform
        with open(mapping_path, "r") as f:
            lines = f.read().split("\n")[:-1]
            self.mapping = {}
            self.unmapping = {}
            for line in lines:
                char_class, char_code = line.split()
                self.mapping[int(char_class)] = chr(int(char_code))
                self.unmapping[chr(int(char_code))] = int(char_class)
        if limit:
            self.dataset = self.dataset.sample(limit)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            images_path = self.dataset.iloc[idx]["path"]
            images = [Image.open(path) for path in images_path]
            if self.transform:
                images = torch.stack([self.transform(np.array(img)) for img in images])  # Apply transform per image
                # images = images.unsqueeze(1)  # Add channel dimension: [Batch, 1, 28, 28]

            labels = [self.unmapping[label] for label in self.dataset.iloc[idx]["label"]]
            labels = torch.tensor(labels, dtype=torch.long)
            return images, labels

        else:
            image_path = self.dataset.iloc[idx]["path"]
            image = Image.open(image_path)
            if self.transform:
                image = np.array(image)
                image = self.transform(image)
                # image = image.unsqueeze(0) 
                # print_tensor_stats(image, "Transformed Image")

            label = self.unmapping[self.dataset.iloc[idx]["label"]]
            label = torch.tensor(label, dtype=torch.long)
            return image, label

augmented_gan_transform = transforms.Compose([
        transforms.Lambda(augmentation_transform),
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize to [-1, 1]
    ])

gan_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize to [-1, 1]
    ])