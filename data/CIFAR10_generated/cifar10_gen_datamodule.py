from data.CIFAR10_generated.cifar10_gen_dataset import CIFAR10GeneratedDataset

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from os.path import expanduser
from data import DataModule


HOME = expanduser("~")

class CIFAR10Generated(DataModule):

    def __init__(self, batch_size, root_dir=None):
            self.batch_size = batch_size
            self._log_hyperparams = False

            if root_dir is None:
                self.root_dir = HOME+"/datasets/cifar10_generated"
            else:
                self.root_dir = root_dir

    def prepare_data(self) -> None:
        CIFAR10GeneratedDataset(root=self.root_dir)

    def prepare_data_per_node(self):
        CIFAR10GeneratedDataset(root=self.root_dir)

    def setup(self, stage = None, split_amount=None, img_size=32) -> None:
        transform = transforms.Compose([
                                        transforms.RandomResizedCrop(img_size),
                                        transforms.RandomHorizontalFlip(),
                                        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        cifar10_set = CIFAR10GeneratedDataset(root=self.root_dir, transforms=transform)
        self.cifar10_train, self.cifar10_test = random_split(cifar10_set, [len(cifar10_set) - 10000, 10000])
        # self.cifar10_train, self.cifar10_val = random_split(cifar10_set, [len(cifar10_set) - 10000, 10000])
        if split_amount is not None:
                self.cifar10_train, _ = random_split(self.cifar10_train, [round(len(self.cifar10_train)*split_amount), len(self.cifar10_train)- round(len(self.cifar10_train)*split_amount)])

    def train_dataloader(self) -> DataLoader:
        cifar10_set = DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True)
        return cifar10_set

    def val_dataloader(self) -> DataLoader:
        satpretrain_val = DataLoader(self.cifar10_val, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True)
        return satpretrain_val

    def test_dataloader(self) -> DataLoader:
        satpretrain_test = DataLoader(self.cifar10_test, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        return satpretrain_test

    @property
    def n_classes(self):
        return 10

    @property
    def n_channels(self):
        return 3

    @property
    def img_size(self):
        return 32

    @property
    def name(self):
        return "CIFAR10_Generated"