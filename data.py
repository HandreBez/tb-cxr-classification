import random

import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image


def collect_images(folder_path):

    image_list = []


    for file_path in folder_path.glob("*.png"):
        if file_path.stem[-1] == '1' :
            tb = 1
        else:
            tb = 0
        image_data = (file_path, tb)
        image_list.append(image_data)

    return image_list


shenzhen_images = collect_images(Path("data/Shenzhen/images/images"))
montgomery_images = collect_images(Path("data/Montgomery/images/images"))

class TBXrayDataset(Dataset):

    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        sample = self.image_list[idx]
        path, label = sample

        image = Image.open(path).convert("L")

        if self.transform is not None:

            image_transformed = self.transform(image)
            return image_transformed,label
        else:
            return image, label
        
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomCrop(224, padding=12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),                
        transforms.Normalize((0.5,), (0.5,))
        ])

    test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),                
            transforms.Normalize((0.5,), (0.5,))
            ])

    return train_transform, test_transform
    
all_images = shenzhen_images + montgomery_images

def get_dataloaders(all_images, batch_size=32, train_ratio=0.9, num_workers=2):
    train_size = int(len(all_images) * train_ratio)
    test_size = len(all_images) - train_size

    train_transform, test_transform = get_transforms()
    
    random.shuffle(all_images)

    train_images = all_images[:train_size]
    test_images = all_images[train_size:]

    train_set = TBXrayDataset(train_images, train_transform)
    test_set = TBXrayDataset(test_images, test_transform)  


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

if __name__ == "__main__":
    train_transform = get_transforms()
    train_loader, test_loader = get_dataloaders(all_images, train_transform)

    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)
    print("Labels shape:", labels.shape)
    print(len(all_images))

