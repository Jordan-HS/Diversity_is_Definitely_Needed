from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from os.path import expanduser
from data import DataModule
import numpy as np
from torch.utils.data import Subset
import random
from data.CIFAR10_generated.cifar10_gen_dataset import CIFAR10GeneratedDataset

from data.few_shot_dataset import FewShotDataset

HOME = expanduser("~")

class CIFAR10DataModule(DataModule):

    def __init__(self, batch_size, root_dir=None, cats_vs_dogs=False):
            self.batch_size = batch_size
            self._log_hyperparams = False

            if root_dir is None:
                self.root_dir = HOME+"/datasets/cifar10"
            else:
                self.root_dir = root_dir

            self.cats_vs_dogs = cats_vs_dogs

    def prepare_data(self) -> None:
        CIFAR10(root=self.root_dir, train=True, download=True)
        CIFAR10(root=self.root_dir, train=False, download=True)

    def prepare_data_per_node(self):
        CIFAR10(root=self.root_dir, train=True, download=True)
        CIFAR10(root=self.root_dir, train=False, download=True)

    def setup(self, stage = None, fed_no=0, dist=None, split_amount=None, img_size=32, num_images=None) -> None:
        if stage in (None, "fit"):
            transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.RandomResizedCrop(img_size),
                                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            cifar10_train = CIFAR10(root=self.root_dir, train=True, download=True, transform=transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_train, [len(cifar10_train) - 10000, 10000])
            if split_amount is not None:
                self.cifar10_train, _ = random_split(self.cifar10_train, [round(len(self.cifar10_train)*split_amount), len(self.cifar10_train)- round(len(self.cifar10_train)*split_amount)])

        elif stage in (None, "few shot"):
            transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.RandomResizedCrop(img_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            assert(num_images is not None, "Please specify a number of images")
            
            dataset = CIFAR10(root=self.root_dir, train=True, download=True)
            
            sampled_data = {}
            for data in dataset:
                if data[1] not in sampled_data:
                    sampled_data[data[1]] = [data[0]]
                elif data[1] in sampled_data and len(sampled_data[data[1]]) < num_images:
                    sampled_data[data[1]].append(data[0])

            self.cifar10_train = FewShotDataset(sampled_data, transforms=transform)
            
        elif stage in (None, "test"):
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(img_size),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            self.cifar10_test = CIFAR10(root=self.root_dir, train=False, download=True, transform=transform)

        elif stage in (None, "federated"):
            assert fed_no > 0, "Federated number must be greater than 0"

            transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.RandomResizedCrop(img_size),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            cifar10_train = CIFAR10(root=self.root_dir, train=True, download=True, transform=transform)

            self.cifar10_train, self.cifar10_val = random_split(cifar10_train, [len(cifar10_train) - 10000, 10000])

            
            fed_train_sets = []

            # Random Split code
            if dist == "iid":
                split_amount = len(cifar10_train) // fed_no
                _temp = cifar10_train
                for _ in range(fed_no):
                    _temp, fed_set = random_split(_temp, [len(_temp) - split_amount, split_amount])
                    fed_train_sets.append(fed_set)
            elif dist == "no-overlap":

                # Non-IID code
                print("Creating non-iid data split...")
                split_amount = len(self.cifar10_train) // fed_no
                classes = round(split_amount / len(self.cifar10_train) * 10)

                unclaimed_classes = [x for x in range(10)]

                fed_train_sets = [] 
                clients_classes = []
                for client in range(fed_no):
                    client_classes = []
                    _classes = []
                    for _ in range(classes):
                            chosen_class = unclaimed_classes.pop(unclaimed_classes.index(random.choice(unclaimed_classes)))
                            idxs = np.nonzero(np.array(cifar10_train.targets) == chosen_class)
                            _classes.append(chosen_class)
                            client_classes += idxs[0].tolist()
                    fed_train_sets.append(Subset(cifar10_train, client_classes))
                    clients_classes.append(_classes)
    
            elif dist == "non-iid":
                fed_train_sets = []
                _temp = cifar10_train
                for _ in range(fed_no):
                    split_amount = random.randint(100, 700)

                    _temp, fed_set = random_split(_temp, [len(_temp) - split_amount, split_amount])
                    fed_train_sets.append(fed_set)

                    

            elif dist == "random-classes":
                split_amount = len(self.cifar10_train) // fed_no
                classes = round(split_amount / len(self.cifar10_train) * 10)

                unclaimed_classes = [x for x in range(10)]

                fed_train_sets = []
                for client in range(fed_no):
                    client_classes = []
                    for _ in range(classes):
                        client_classes.append(random.choice(unclaimed_classes))
                    print(client_classes)
                    fed_train_sets.append(Subset(cifar10_train, client_classes))
                

            self.fed_train_sets = fed_train_sets

            return clients_classes
            

    def train_dataloader(self) -> DataLoader:
        cifar10_train = DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True)
        return cifar10_train

    def fed_train_dataloader(self):
        loaders = []
        for train_set in self.fed_train_sets:
            loaders.append(DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=16))
        return loaders

    def val_dataloader(self) -> DataLoader:
        cifar10_val = DataLoader(self.cifar10_val, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        return cifar10_val

    def test_dataloader(self) -> DataLoader:
        cifar10_test = DataLoader(self.cifar10_test, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        return cifar10_test

    @property
    def n_classes(self):
        if self.cats_vs_dogs:
            return 2
        return 10

    @property
    def n_channels(self):
        return 3

    @property
    def img_size(self):
        return 32

    @property
    def name(self):
        return "CIFAR10"

    def random_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        rand_train_split, _ = random_split(self.cifar10_train, [len(self.cifar10_train) - n_images, n_images])
        loader = DataLoader(rand_train_split, batch_size=batch_size, num_workers=num_worker)
        return loader
