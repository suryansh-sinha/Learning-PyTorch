"""
Contains functionality for creating a PyTorch DataLoaders for image classification
data.
"""

import os

from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
  train_dir: str,
  test_dir: str,
  transform: transforms.Compose,
  batch_size: int,
  num_workers: int = NUM_WORKERS
):
  """ Creates training and testing DataLoaders

  Takes in a training and testing directory and creates PyTorch Datasets and DataLoaders

  Args:
    train_dir: Path to training directory
    test_dir: Path to testing directory
    transform: torchvision transforms to perform on training and testing data
    batch_size: number of samples in each batch of the dataloaders
    num_workers: Number of CPU cores that load the data

  Returns:
     A tuple of (train_dataloader, test_dataloader, class_names)
     Where class_names is a list of target classes.

  Example usage:
    train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir,
      test_dir=path/to/test_dir,
      transform=some_transform,
      batch_size=32,
      num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into DataLoaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
  )

  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
  )

  return train_dataloader, test_dataloader, class_names
