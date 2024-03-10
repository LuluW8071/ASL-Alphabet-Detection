
import os 
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

def create_dataloaders(source_dir, transform, batch_size, num_workers):
    # Use ImageFolder to create datasets
    source_data = datasets.ImageFolder(root = source_dir,
                                       transform = transform) # Transforms input data into tensors

    # Get the class names
    class_names = source_data.classes

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(source_data))
    test_size = len(source_data) - train_size
    train_data, test_data = random_split(source_data, [train_size, test_size])

    # Turn images into dataloaders
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_data, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 num_workers=num_workers, 
                                 pin_memory=True)

    return train_dataloader, test_dataloader, class_names