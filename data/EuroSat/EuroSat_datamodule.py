from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from os.path import expanduser
from data.few_shot_dataset import FewShotDataset
HOME = expanduser("~")

class EuroSATDatamodule:
    def __init__(self, batch_size, root_dir=None):
            self.batch_size = batch_size

            if root_dir is None:
                self.root_dir = HOME+"/datasets/EuroSAT"
            else:
                self.root_dir = root_dir
                
    def setup(self, stage = None, split_amount=None, img_size=64) -> None:

        if stage == "fit":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomRotation(90),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Resize(256),
                                            transforms.RandomCrop(224)])
            self.eurosat_train = EuroSAT(root=self.root_dir, download=False, transform=transform)
        
        
        elif stage == "test":

            test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize(256),
                                            transforms.CenterCrop(224)])
            self.eurosat_test = EuroSAT(root=self.root_dir, download=True, transform=test_transform)
            
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.eurosat_train, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True)
            
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.eurosat_test, batch_size=self.batch_size, num_workers=16, pin_memory=True, shuffle=True)

    @property
    def n_classes(self):
        return 10
    
    @property
    def n_channels(self):
        return 3
    
    @property
    def name(self):
        return "EuroSAT"