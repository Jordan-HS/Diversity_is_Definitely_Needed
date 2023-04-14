import random
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from os.path import expanduser
from data import DataModule
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
import numpy as np

from data.few_shot_dataset import FewShotDataset


HOME = expanduser("~")

class CIFAR100DataModule(DataModule):

    def __init__(self, batch_size, root_dir=None):
            self.batch_size = batch_size
            self._log_hyperparams = False

            if root_dir is None:
                self.root_dir = HOME+"/datasets/cifar100"
            else:
                self.root_dir = root_dir

    def prepare_data(self) -> None:
        CIFAR100(root=self.root_dir, train=True, download=True)
        CIFAR100(root=self.root_dir, train=False, download=True)

    def prepare_data_per_node(self):
        CIFAR100(root=self.root_dir, train=True, download=True)
        CIFAR100(root=self.root_dir, train=False, download=True)

    def prepare_ddp(self, rank, world_size, pin_memory=False, num_workers=0):
        train_sampler = DistributedSampler(self.cifar100_train, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        dataloader = DataLoader(self.cifar100_train, batch_size=self.batch_size, num_workers=num_workers,
                                drop_last=False, shuffle=False, sampler=train_sampler)
        return dataloader

    def setup(self, stage = None, fed_no=0, dist=None, split_amount=None, img_size=32, num_images=None) -> None:
        if stage in (None, "fit"):
            transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.RandomResizedCrop(img_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            cifar100_train = CIFAR100(root=self.root_dir, train=True, download=True, transform=transform)
            self.cifar100_train, self.cifar100_val = random_split(cifar100_train, [len(cifar100_train) - 10000, 10000])
            if split_amount is not None:
                self.cifar100_train, _ = random_split(self.cifar100_train, [round(len(self.cifar100_train)*split_amount), len(self.cifar100_train)- round(len(self.cifar100_train)*split_amount)])

        elif stage in (None, "few shot"):
            transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.RandomResizedCrop(img_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            assert(num_images is not None, "Please specify a number of images")
            
            dataset = CIFAR100(root=self.root_dir, train=True, download=True)
            
            sampled_data = {}
            for data in dataset:
                if data[1] not in sampled_data:
                    sampled_data[data[1]] = [data[0]]
                elif data[1] in sampled_data and len(sampled_data[data[1]]) < num_images:
                    sampled_data[data[1]].append(data[0])

            self.cifar100_train = FewShotDataset(sampled_data, transforms=transform)
        
            
        elif stage in (None, "test"):
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(img_size),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            self.cifar100_test = CIFAR100(root=self.root_dir, train=False, download=True, transform=transform)

        elif stage in (None, "federated"):
            assert fed_no > 0, "Federated number must be greater than 0"

            transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.RandomResizedCrop(img_size),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            cifar100_train = CIFAR100(root=self.root_dir, train=True, download=True, transform=transform)

            self.cifar100_train, self.cifar100_val = random_split(cifar100_train, [len(cifar100_train) - 10000, 10000])
            
            

            
            fed_train_sets = []

            # Random Split code
            if dist == "iid":
                split_amount = len(cifar100_train) // fed_no
                _temp = cifar100_train
                for _ in range(fed_no):
                    _temp, fed_set = random_split(_temp, [len(_temp) - split_amount, split_amount])
                    fed_train_sets.append(fed_set)
            elif dist == "no-overlap":

                # Non-IID code
                print("Creating non-iid data split...")
                split_amount = len(self.cifar100_train) // fed_no
                classes = round(split_amount / len(self.cifar100_train) * 100)

                unclaimed_classes = [x for x in range(100)]

                fed_train_sets = [] 
                clients_classes = []
                for client in range(fed_no):
                    client_classes = []
                    _classes = []
                    for _ in range(classes):
                            chosen_class = unclaimed_classes.pop(unclaimed_classes.index(random.choice(unclaimed_classes)))
                            idxs = np.nonzero(np.array(cifar100_train.targets) == chosen_class)
                            _classes.append(chosen_class)
                            client_classes += idxs[0].tolist()
                    fed_train_sets.append(Subset(cifar100_train, client_classes))
                    clients_classes.append(_classes)
    
            elif dist == "non-iid":
                fed_train_sets = []
                _temp = cifar100_train
                for _ in range(fed_no):
                    split_amount = random.randint(100, 700)

                    _temp, fed_set = random_split(_temp, [len(_temp) - split_amount, split_amount])
                    fed_train_sets.append(fed_set)

                

            elif dist == "random-classes":
                split_amount = len(self.cifar100_train) // fed_no
                classes = round(split_amount / len(self.cifar100_train) * 100)

                unclaimed_classes = [x for x in range(100)]

                fed_train_sets = []
                for client in range(fed_no):
                    client_classes = []
                    for _ in range(classes):
                        client_classes.append(random.choice(unclaimed_classes))
                    print(client_classes)
                    fed_train_sets.append(Subset(cifar100_train, client_classes))
                

            self.fed_train_sets = fed_train_sets

            return clients_classes

    def train_dataloader(self) -> DataLoader:
        cifar100_train = DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True)
        return cifar100_train

    def fed_train_dataloader(self):
        loaders = []
        for train_set in self.fed_train_sets:
            loaders.append(DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=16))
        return loaders

    def val_dataloader(self) -> DataLoader:
        cifar100_val = DataLoader(self.cifar100_val, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        return cifar100_val

    def test_dataloader(self) -> DataLoader:
        cifar100_test = DataLoader(self.cifar100_test, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        return cifar100_test

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
        return "CIFAR100"

    def random_sub_train_loader(self, n_images, batch_size, num_worker=1, num_replicas=None, rank=None):
        rand_train_split, _ = random_split(self.cifar100_train, [len(self.cifar100_train) - n_images, n_images])
        loader = DataLoader(rand_train_split, batch_size=batch_size, num_workers=num_worker)
        return loader
