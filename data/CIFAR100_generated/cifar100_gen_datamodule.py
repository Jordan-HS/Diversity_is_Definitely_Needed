from data.CIFAR100_generated.cifar100_gen_dataset import CIFAR100GeneratedDataset

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from os.path import expanduser
from data import DataModule


HOME = expanduser("~")

class CIFAR100Generated(DataModule):

    def __init__(self, batch_size, root_dir=None):
            self.batch_size = batch_size
            self._log_hyperparams = False

            if root_dir is None:
                self.root_dir = HOME+"/datasets/cifar100_generated"
            else:
                self.root_dir = root_dir

    def prepare_data(self) -> None:
        CIFAR100GeneratedDataset(root=self.root_dir)

    def prepare_data_per_node(self):
        CIFAR100GeneratedDataset(root=self.root_dir)

    def setup(self, stage = None, fed_no = 0, split_amount=None, img_size=32) -> None:
        transform = transforms.Compose([
                                        transforms.RandomResizedCrop(img_size),
                                        transforms.RandomHorizontalFlip(),
                                        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        cifar100_set = CIFAR100GeneratedDataset(root=self.root_dir, transforms=transform)
        self.cifar100_train, self.cifar100_test = random_split(cifar100_set, [len(cifar100_set) - 10000, 10000])
        # self.cifar100_train, self.cifar100_val = random_split(cifar100_set, [len(cifar100_set) - 10000, 10000])
        
        if split_amount is not None:
            self.cifar100_train, _ = random_split(self.cifar100_train, [round(len(self.cifar100_train)*split_amount), len(self.cifar100_train)- round(len(self.cifar100_train)*split_amount)])


    def train_dataloader(self) -> DataLoader:
        cifar100_set = DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True)
        return cifar100_set

    def val_dataloader(self) -> DataLoader:
        satpretrain_val = DataLoader(self.cifar100_val, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True)
        return satpretrain_val

    def test_dataloader(self) -> DataLoader:

        satpretrain_test = DataLoader(self.cifar100_test, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        return satpretrain_test

    @property
    def n_classes(self):
        return 100

    @property
    def n_channels(self):
        return 3

    @property
    def img_size(self):
        return 32

    @property
    def name(self):
        return "CIFAR100_Generated"