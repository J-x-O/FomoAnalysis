import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import save_image

from networks.DDAMFNpp_affectnet7 import DDAMFNppAffectnet7
from networks.DDAMFNpp_rafdb import DDAMFNppRAFDB
from src.Consts import classes
from src.RetinaFaceAlign import transform_stack
from src.VideoUtil import FrameIterator, VideoTarget


def test_full():
    tensor = test_img()
    output = test_model(tensor)
    plot_results(output)


def test_img():
    img = cv2.imread("data/test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("test shape:", img.shape)
    face_tensor = transform_stack(img)
    print("test cropped shape:", face_tensor.shape)
    save_image(face_tensor, "data/test_transformed.jpg")
    return face_tensor

def test_video():
    for index, frame in FrameIterator(VideoTarget("data/survey/5fd09851-a593-4bea-80f1-322004743c44", "Disgust_3_0876.webm")):
        face_tensor = transform_stack(frame)
        save_image(face_tensor, f"data/test_transformed_{index}.jpg")
        if index > 5:
            break


def test_model(img: Image):
    model1 = DDAMFNppRAFDB()
    model2 = DDAMFNppAffectnet7()
    with torch.no_grad():
        outputs1, _, _ = model1(img)
        outputs2, _, _ = model2(img)
    return outputs1


def plot_results(outputs: torch.Tensor):
    probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(classes)), probabilities[0], align='center', alpha=0.7)
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.title('Emotion Prediction Probabilities')
    plt.show()