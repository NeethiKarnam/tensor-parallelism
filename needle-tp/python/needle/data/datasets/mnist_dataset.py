from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename) as image_file:
          _, num_examples, num_rows, num_cols = struct.unpack('>IIII', image_file.read(16))
          input_dim = num_rows * num_cols
          X = np.frombuffer(image_file.read(), dtype=np.uint8).reshape(num_examples, input_dim).astype(np.float32) / 255.0
        
        with gzip.open(label_filename) as label_file:
          _, num_labels = struct.unpack('>II', label_file.read(8))
          y = np.frombuffer(label_file.read(), dtype=np.uint8)

        self.images = X
        self.labels = y
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X = self.images[index]
        y = self.labels[index]

        if isinstance(index, (int, np.integer)):
          X = X.reshape(self.num_rows, self.num_cols, 1)
          if self.transforms is not None:
              for transform in self.transforms:
                X = transform(X)
        else:
          X = X.reshape(-1, self.num_rows, self.num_cols, 1)
          if self.transforms is not None:
            for transform in self.transforms:
              X = np.stack([transform(img) for img in X], axis=0)

        return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION