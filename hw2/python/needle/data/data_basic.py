import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        # Reset batch counter at the start of each epoch
        self.batch_idx = 0
        
        # If shuffle is True, create a new random ordering for this epoch
        if self.shuffle:
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            self.ordering = np.array_split(indices, 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        # Otherwise, use the pre-computed ordering from __init__
        return self

    def __next__(self):
        # Check if we've exhausted all batches
        if self.batch_idx >= len(self.ordering):
            raise StopIteration
        
        # Get the indices for the current batch
        batch_indices = self.ordering[self.batch_idx]
        self.batch_idx += 1
        
        # Fetch all samples in this batch from the dataset
        batch = [self.dataset[i] for i in batch_indices]
        
        # Handle datasets that return tuples of any length
        # For example: (features,) or (features, labels) or more
        # Stack each element of the tuple separately
        if isinstance(batch[0], tuple):
            # Number of elements in each sample
            num_elements = len(batch[0])
            result = []
            for elem_idx in range(num_elements):
                stacked = np.array([item[elem_idx] for item in batch])
                result.append(Tensor(stacked))
            return tuple(result)
        else:
            # If dataset returns single values (not tuples)
            return Tensor(np.array(batch))

