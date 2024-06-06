from typing import Union

import numpy as np
import torch
from deepface import DeepFace
import cv2
from PIL import Image
from torchvision import transforms


class RetinaFaceAlign:
    def __init__(self, detector_backend='retinaface'):
        self.detector_backend = detector_backend

    def __call__(self, img):

        if isinstance(img, str):
            img = Image.open(img)
        if isinstance(img, Image.Image):
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Extract faces using DeepFace
        face_objs = DeepFace.extract_faces(
            img_path=img,
            detector_backend=self.detector_backend,
            enforce_detection=False
        )

        if not face_objs:
            raise ValueError("No faces detected.")

        # Find the face with the highest confidence
        most_confident_face = max(face_objs, key=lambda face: face["confidence"])
        face_box = most_confident_face['facial_area']
        x, y, w, h = face_box['x'], face_box['y'], face_box['w'], face_box['h']
        face_crop = img[y:y + h, x:x + w]

        return Image.fromarray(face_crop)


def transform_stack(img: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
    tensor = transforms.Compose([
        RetinaFaceAlign(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(img)
    tensor = tensor.unsqueeze(0)
    return tensor
