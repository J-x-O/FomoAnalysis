import sys

import torch
from networks.DDAM import DDAMNet
from torchvision import transforms
from PIL import Image


def do_stuff():
    # Define the model

    model = DDAMNet(num_class=7, num_head=2, pretrained=False)

    checkpoint_path = 'networks/DDAMFNpp/pretrained/MFN_msceleb.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
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
    img = img.unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        outputs, _, _ = model(img)
        _, predicted = torch.max(outputs, 1)
        print(f'Predicted class: {predicted.item()}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    do_stuff()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
