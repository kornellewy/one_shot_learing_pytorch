import torch
import torch.nn as nn
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset_multiple_classes import DatasetMultipleClasses


class DatasetMultiple(pl.LightningDataModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self._hparams = hparams
        self.img_transform = A.Compose(
            [
                A.Resize(100, 100),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
                ),
                A.PadIfNeeded(
                    min_height=100, min_width=100, always_apply=True, border_mode=0
                ),
                A.IAAAdditiveGaussianNoise(p=0.1),
                A.IAAPerspective(p=0.1),
                A.RandomBrightnessContrast(p=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def setup(self, stage):
        dataset = DatasetMultipleClasses(
            self._hparams["dataset_path"], img_transform=self.img_transform
        )
        print("len(dataset): ", len(dataset))
        if len(dataset) > 2000:
            train_size = int(0.95 * len(dataset))
            valid_size = int(0.05 * len(dataset))
        else:
            train_size = int(0.95 * len(dataset))
            valid_size = int(0.05 * len(dataset))
        rest = len(dataset) - train_size - valid_size
        train_size = train_size + rest
        self.train_set, self.valid_set = torch.utils.data.random_split(
            dataset, [train_size, valid_size]
        )
        self.test_set = DatasetMultipleClasses(
            self._hparams["test_dataset_path"], img_transform=self.img_transform
        )

            
            
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self._hparams["batch_size"]
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_set, batch_size=self._hparams["batch_size"]
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set, batch_size=self._hparams["batch_size"]
        )
