from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
from apps.simple_ml import parse_mnist

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.transforms = transforms
        self.images, self.labels = parse_mnist(image_filename, label_filename)

    def __getitem__(self, index) -> object:
        # Handle slicing
        if isinstance(index, slice):
            # For slicing, return stacked arrays
            indices = range(*index.indices(len(self)))
            images_list = []
            labels_list = []
            for i in indices:
                img, lbl = self[i]  # Recursively call with single index
                images_list.append(img)
                labels_list.append(lbl)
            return np.array(images_list), np.array(labels_list)
        
        # Handle single index
        # parse_mnist returns flattened images (784,), need to reshape to (28, 28)
        image = self.images[index].reshape(28, 28, 1)  # Add channel dimension for transforms
        label = self.labels[index]
        
        # Apply transforms if any (transforms expect H x W x C format)
        if self.transforms is not None:
            for transform in self.transforms:
                image = transform(image)
        
        # Return image as (C, H, W) format and label
        # Transpose from (H, W, C) to (C, H, W)
        return np.transpose(image, (2, 0, 1)), label

    def __len__(self) -> int:
        return len(self.images)