from torch.utils.data import DataLoader


class DataModule():

    def prepare_data(self) -> None:
        raise NotImplementedError

    def setup(self, stage = None, fed_no=0) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @property
    def n_classes(self):
        raise NotImplementedError

    @property
    def n_channels(self):
        raise NotImplementedError

    @property
    def img_size(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    def random_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        raise NotImplementedError
