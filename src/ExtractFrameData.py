import json

import cv2
import torch

from networks.DDAMFNpp_affectnet7 import DDAMFNppAffectnet7
from networks.DDAMFNpp_rafdb import DDAMFNppRAFDB
from src.RetinaFaceAlign import transform_stack


class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']


def write_frame_data(workdir: str, video_name: str):
    vidcap = cv2.VideoCapture(f"{workdir}/{video_name}")
    models = {
        "affectnet7": DDAMFNppAffectnet7(),
        "rafdb": DDAMFNppRAFDB()
    }
    data = {
        "affectnet7": [],
        "rafdb": []
    }
    success, image = vidcap.read()
    count = 0
    while success:
        print(f"--- {count} ---")
        cropped = transform_stack(image)
        for model_name, model in models.items():
            with torch.no_grad():
                outputs, _, _ = model(cropped)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            data[model_name].append(probabilities[0])
        success, image = vidcap.read()
        count += 1
    for model_name, model_data in data.items():
        with open(f"{workdir}/{video_name}_{model_name}.json", "w") as f:
            f.write(json.dumps(model_data))
