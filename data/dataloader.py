import torch
from torch.utils.data import random_split
from data.dataset import EtMirADatasetTraining, EtMirADatasetValidation
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from config import TRAIN_PATH, VAL_PATH


class EtMirADataLoader(pl.LightningDataModule):
    def __init__(self, transform_input, transform_gt, use_augment, batch_size_train, batch_size_val, patch_size) -> None:
        super().__init__()
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_val
        self.num_workers = 32
        evaluation_dataset = EtMirADatasetValidation(
            data_path=VAL_PATH, transform_input=transform_input, transform_gt=transform_gt, patch_size=patch_size)

        # Set the seed for the random number generator
        torch.manual_seed(0)
        # Calculate the lengths of the validation and test datasets
        val_len = int(len(evaluation_dataset) * 0.5)  # 50% of the data
        test_len = len(evaluation_dataset) - val_len
        # Split the evaluation dataset
        self.val_dataset, self.test_dataset = random_split(
            evaluation_dataset, [val_len, test_len])

        self.train_dataset = EtMirADatasetTraining(
            data_path=TRAIN_PATH, transform_input=transform_input, transform_gt=transform_gt, use_augment=use_augment, patch_sizes=patch_size)

    def get_train_dataloader(self):
        dataset = DataLoader(self.train_dataset, batch_size=self.batch_size_train, prefetch_factor=5, num_workers=self.num_workers,
                             shuffle=True, drop_last=True, pin_memory=True)
        return dataset

    def get_val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size_val, num_workers=self.num_workers, shuffle=False)

    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_test, num_workers=self.num_workers, shuffle=False)
