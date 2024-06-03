import torch
from torch import nn

from networks.MixedFeatureNet import MixedFeatureNet
from networks.DDAM import DDAMNet
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
    # Define the model

    checkpoint_path = 'networks/DDAMFNpp/pretrained/MFN_msceleb.pth'
    model = UpdatedDDAMNet(num_class=7, num_head=2, pretrained_path=checkpoint_path)
    model.eval()

    # Define the image transformations
    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess the image
    img_path = 'data/test.jpg'
    img = Image.open(img_path)
    img = data_transforms(img)
    save_image(img, "data/test_transformed.jpg")
    img = img.unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        outputs, _, _ = model(img)

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
