import cv2


def count_frames(path: str):
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        count += 1
    print(f"Total frames: {count}")


if __name__ == '__main__':
    count_frames("../data/survey/1c069c5e-82f5-46a0-80eb-b9daa63bd3b2/Anger_8_0971.webm")