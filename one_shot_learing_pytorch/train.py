"""
source: 
https://github.com/cskarthik7/One-Shot-Learning-PyTorch

"""
from pathlib import Path

import pytorch_lightning as pl

from datamodule import DatasetMultiple
from siamese_module import SiameseModule

if __name__ == "__main__":
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_loss",
        dirpath="one_shot_model/",
        filename="{epoch:02d}-{valid_loss:.2f}",
        save_top_k=3,
        mode="min",
        save_last=True,
        every_n_epochs=1,
    )
    hparams = {
        "batch_size": 256,
        "lr": 0.00006,
        "dataset_path": "J:/deepcloth/datasets/one_shot_dataset/train",
        "test_dataset_path": "J:/deepcloth/datasets/one_shot_dataset/test",
    }
    model_module = SiameseModule(hparams=hparams)
    data_module = DatasetMultiple(hparams=hparams)
    checkpoint_save_path = str(Path(__file__).parent)
    # checkpoint_path = "one_shot_model/last.ckpt"

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        benchmark=True,
        max_epochs=2000,
        default_root_dir=checkpoint_save_path,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        # resume_from_checkpoint=checkpoint_path,
    )
    # lr_finder = trainer.tuner.lr_find(model_module, data_module)
    # print(lr_finder.results)
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # new_lr = lr_finder.suggestion()
    # print(new_lr)

    trainer.fit(model_module, data_module)
    trainer.test(model_module, data_module)
