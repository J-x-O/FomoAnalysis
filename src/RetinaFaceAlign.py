import numpy as np
from deepface import DeepFace
import cv2
from PIL import Image
from torchvision import transforms


class RetinaFaceAlign:
    def __init__(self, detector_backend='retinaface'):
        self.detector_backend = detector_backend

    def __call__(self, img):
        # Convert PIL image to numpy array
        img_array = np.array(img)
        print(img_array.shape)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Extract faces using DeepFace
        face_objs = DeepFace.extract_faces(
            img_path=img_rgb,
            detector_backend=self.detector_backend,
            enforce_detection=False
        )

        if not face_objs:
            raise ValueError("No faces detected.")

        # Find the face with the highest confidence
        most_confident_face = max(face_objs, key=lambda face: face["confidence"])
        face_box = most_confident_face['facial_area']
        x, y, w, h = face_box['x'], face_box['y'], face_box['w'], face_box['h']
        face_crop = img_array[y:y + h, x:x + w]

        return Image.fromarray(face_crop)


transform_stack = transforms.Compose([
    RetinaFaceAlign(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])