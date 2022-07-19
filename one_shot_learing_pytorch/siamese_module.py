"""
source: 
https://github.com/cskarthik7/One-Shot-Learning-PyTorch

"""
from pathlib import Path

import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score

from dataset_multiple_classes import DatasetMultipleClasses
from model import Siamese


class SiameseModule(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
    ):
        super().__init__()
        self._hparams = hparams
        self.model = Siamese()
        self.criterion = torch.nn.BCEWithLogitsLoss(size_average=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"])
        return optimizer

    def forward(self, img0, img1):
        return self.model(img0, img1)

    def training_step(self, batch, _):
        img0, img1, label = batch
        pred, _ = self(img0, img1)
        loss = self.criterion(pred, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        img0, img1, label = batch
        pred, _ = self(img0, img1)
        loss = self.criterion(pred, label)
        self.log("valid_loss", loss)
        return pred, label

    def validation_epoch_end(self, batches):
        pred_list = []
        label_list = []
        for batch in batches:
            preds, labels = batch
            preds = preds.data.cpu().numpy()
            preds = preds.squeeze(1).tolist()
            labels = labels.data.cpu().numpy()
            labels = labels.squeeze(1).tolist()
            preds = [1 if pred > 0 else 0 for pred in preds]
            labels = [1 if label > 0 else 0 for label in labels]
            pred_list += preds
            label_list += labels
        acc = accuracy_score(label_list, pred_list)
        self.log("valid_acc", acc)

    def test_step(self, batch, _):
        img0, img1, label = batch
        pred, _ = self(img0, img1)
        return pred, label

    def test_epoch_end(self, batches):
        pred_list = []
        label_list = []
        for batch in batches:
            preds, labels = batch
            preds = preds.data.cpu().numpy()
            preds = preds.squeeze(1).tolist()
            labels = labels.data.cpu().numpy()
            labels = labels.squeeze(1).tolist()
            preds = [1 if pred > 0 else 0 for pred in preds]
            labels = [1 if label > 0 else 0 for label in labels]
            pred_list += preds
            label_list += labels
        acc = accuracy_score(label_list, pred_list)
        self.log("test_acc", acc)


if __name__ == "__main__":
    hparams = {
        "batch_size": 256,
        "lr": 0.00006,
        "dataset_path": "J:/yt_image_dataset_maker/face_dataset_train",
        "test_dataset_path": "J:/yt_image_dataset_maker/face_dataset_test",
    }
    kjn = SiameseModule(hparams=hparams)
    checkpoint_save_path = str(Path(__file__).parent)
    checkpoint_path = ""

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        benchmark=True,
        max_epochs=200,
        default_root_dir=checkpoint_save_path,
        check_val_every_n_epoch=1,
        resume_from_checkpoint=checkpoint_path,
    )
    trainer.fit(kjn)
    trainer.test(kjn)
