import cv2
import torch
from deepface import DeepFace
from torch import nn

from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from networks.DDAM import DDAMNet
from networks.MixedFeatureNet import MixedFeatureNet


class UpdatedDDAMNet(DDAMNet):
    def __init__(self, num_class=7, num_head=2, pretrained_path=None):
        super(UpdatedDDAMNet, self).__init__(num_class=num_class, num_head=num_head, pretrained=False)
        if pretrained_path:
            # Load the pretrained MixedFeatureNet model
            mixed_feature_net = torch.load(pretrained_path, map_location='cpu')
            if isinstance(mixed_feature_net, MixedFeatureNet):
                self.features = nn.Sequential(*list(mixed_feature_net.children())[:-4])
            else:
                raise TypeError("Expected loaded model to be an instance of MixedFeatureNet")


def do_stuff():
    target_image = "data/happy.webp"

    face_objs = DeepFace.extract_faces(
        img_path=target_image,
        detector_backend='retinaface',
    )
    # iterate over faceobjs and find the one with the highest confidence
    most_confident_face = max(face_objs, key=lambda face: face["confidence"])

    face_box = most_confident_face['facial_area']
    x, y, w, h = face_box['x'], face_box['y'], face_box['w'], face_box['h']

    # Load the original image
    img = cv2.imread(target_image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 3: Crop the Most Confident Face
    face_crop = img_rgb[y:y + h, x:x + w]

    # Convert to PIL image for further processing
    face_pil = Image.fromarray(face_crop)

    # Define the image transformations
    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    face_pil = data_transforms(face_pil)
    save_image(face_pil, "data/happy_transformed.jpg")
    face_pil = face_pil.unsqueeze(0)  # Add batch dimension

    checkpoint_path = 'networks/DDAMFNpp/pretrained/MFN_msceleb.pth'
    model = UpdatedDDAMNet(num_class=7, num_head=2, pretrained_path=checkpoint_path)
    model.eval()

    # Run inference
    with torch.no_grad():
        outputs, _, _ = model(face_pil)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

    # Define class names (these should match the order used in your model)
    class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']

    # Plot the probabilities
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(class_names)), probabilities[0], align='center', alpha=0.7)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.title('Emotion Prediction Probabilities')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    do_stuff()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
