import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import save_image

from networks.DDAMFNpp_affectnet7 import DDAMFNppAffectnet7
from networks.DDAMFNpp_rafdb import DDAMFNppRAFDB
from src.Consts import classes
from src.RetinaFaceAlign import transform_stack


def test_full():
    tensor = test_img()
    output = test_model(tensor)
    plot_results(output)


def test_img():
    img = Image.open("data/test.jpg")
    face_tensor = transform_stack(img)
    face_tensor = face_tensor.unsqueeze(0)
    save_image(face_tensor, "data/test_transformed.jpg")
    return face_tensor


def test_model(img: Image):
    model = DDAMFNppRAFDB()
    with torch.no_grad():
        outputs, _, _ = model(img)
    return outputs


def plot_results(outputs: torch.Tensor):
    probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(classes)), probabilities[0], align='center', alpha=0.7)
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.title('Emotion Prediction Probabilities')
    plt.show()