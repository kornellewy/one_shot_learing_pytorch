from pathlib import Path
from PIL import Image
import os

import cv2
import torch
import numpy as np
from torchvision import transforms
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from siamese_module import SiameseModule
from utils import load_files_with_given_extension


class Comparer:
    def __init__(
        self,
        model_path: Path,
        train_dataset_path: Path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.device = device
        self.img_transforms = transforms.Compose(
            [
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        hparams = {
            "batch_size": 256,
            "lr": 0.00006,
        }
        self.model = (
            SiameseModule(hparams=hparams)
            .load_from_checkpoint(model_path, hparams=hparams)
            .to(self.device)
        )
        self.model.eval()
        self.train_dataset_path = train_dataset_path
        self.classifier = KNeighborsClassifier(n_neighbors=3)
        class_idxs, image_tensors = self.get_base_data_to_fit()
        self.classifier.fit(image_tensors, class_idxs)

    def get_base_data_to_fit(self):
        all_images_paths = load_files_with_given_extension(self.train_dataset_path)
        class_idxs = []
        image_tensors = []
        for image_path in all_images_paths:
            class_idx = Path(image_path).parent.stem
            image_tensor = (
                self.get_image_encoding_tensor(image_path)
                .data.cpu()
                .numpy()
                .squeeze(0)
                .tolist()
            )
            class_idxs.append(int(class_idx))
            image_tensors.append(image_tensor)
        return class_idxs, image_tensors

    def get_image_encoding_tensor(self, image_path: str) -> torch.Tensor:
        image_tensor = self.read_and_transform_image_from_path(image_path)
        with torch.no_grad():
            image_encode_tensor = self.model.model.forward_one(image_tensor)
        return image_encode_tensor

    def read_and_transform_image_from_path(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        image_tensor = self.img_transforms(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        return image_tensor

    def detect_from_path(self, image_path: str):
        image_tensor = self.read_and_transform_image_from_path(image_path)
        # prediction
        with torch.no_grad():
            tensor = self.model.model.forward_one(image_tensor)
            tensor = tensor.data.cpu().numpy().squeeze(0).tolist()
            class_idx = self.classifier.predict([tensor])[0]
        return class_idx

    def detect_from_numpy(self, image: np.ndarray):
        image_tensor = self.read_and_transform_image_from_numpy(image)
        with torch.no_grad():
            tensor = self.model.model.forward_one(image_tensor)
            tensor = tensor.data.cpu().numpy().squeeze(0).tolist()
            class_idx = self.classifier.predict([tensor])[0]
        return class_idx

    def read_and_transform_image_from_numpy(self, image: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image)
        image_tensor = self.img_transforms(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        return image_tensor


if __name__ == "__main__":
    train_dataset_path = Path("wyborowa_one_shot_train")
    test_dataset_path = Path("wyborowa_one_shot_test")
    model_path = Path("one_shot_model\epoch=186-valid_loss=0.27.ckpt")
    device = torch.device("cuda")
    face_comparer = Comparer(
        model_path=model_path, device=device, train_dataset_path=train_dataset_path
    )
    test_images_paths = load_files_with_given_extension(test_dataset_path)
    preditions = []
    lables = []
    for image_path in test_images_paths:
        lable = Path(image_path).parent.stem
        pred = face_comparer.detect_from_path(image_path)
        preditions.append(int(pred))
        lables.append(int(lable))
        print("image_class: ", lable, "pred: ", pred)
    acc = accuracy_score(lables, preditions)
    print("acc", acc)
