import torch
import os
from torchvision.datasets import VisionDataset
import numpy as np
from PIL import Image


class CIFAR10GeneratedDataset(VisionDataset):

    def __init__(self, 
                 root: str,
                 transforms=None, 
                 transform=None, 
                 target_transform=None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.samples = []
        
        self.classes = ['airplane',
                        'automobile',
                        'bird',
                        'cat',
                        'deer',
                        'dog',
                        'frog',
                        'horse',
                        'ship',
                        'truck']

        folders = os.listdir(root)
        for folder in folders:
            _temp_files = []
            for f in os.listdir(os.path.join(root, folder)):
                _temp_files.append(os.path.join(root, folder, f))
            self.samples += _temp_files


    def __getitem__(self, index: int) -> torch.Tensor:
        image = np.array(Image.open(self.samples[index]))
        image = torch.Tensor(image).float()
        image = torch.einsum("hwc->chw", image)[0:3,...]
        # image = image.reshape((3, 256, 256))
        image /= 255


        label = self.classes.index(self.samples[index].split("/")[-2])

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

        
    def __len__(self) -> int:
        return len(self.samples)