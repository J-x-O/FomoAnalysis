import os

import cv2
import numpy as np
from tqdm import tqdm


def count_frames(path: str):
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        count += 1
    return count


class VideoTarget:

    def __init__(self, workdir: str, video_name: str):
        self.workdir = workdir
        self.video_name = video_name
        self.video_title = video_name.split(".")[0]
        self.valid = os.path.exists(f"{workdir}/{video_name}")
        self.total = count_frames(f"{workdir}/{video_name}") if self.valid else 0

    def __str__(self):
        return f"{self.workdir} -> {self.video_name}"

    @property
    def frame_count(self):
        return count_frames(self.full_path)

    @property
    def full_path(self):
        return f"{self.workdir}/{self.video_name}"

    def side_car(self, appendix: str):
        return f"{self.workdir}/{self.video_title}_{appendix}"

    def has_side_car(self, appendix: str):
        return os.path.exists(self.side_car(appendix))

    def has_all_side_cars(self, appendix: list[str]):
        return all([self.has_side_car(a) for a in appendix])


class FrameIterator:
    def __init__(self, video: VideoTarget, console: bool = True):
        self.vidcap = cv2.VideoCapture(video.full_path)
        self.frame = None
        self.frame_index = -1
        self.total = video.frame_count
        self.tqdm = tqdm(total=self.total, desc=f"Extracting {video.video_title}", position=0) if console else None

    def __iter__(self):
        return self

    def __next__(self):
        success, self.frame = self.vidcap.read()
        if not success or np.shape(self.frame) == ():
            self.stop()

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.frame_index += 1
        if self.tqdm: self.tqdm.update(1)
        return self.frame_index, self.frame


    def stop(self):
        if self.tqdm: self.tqdm.close()
        self.vidcap.release()
        raise StopIteration


def find_all_videos(workdir: str) -> list[VideoTarget]:
    result = []
    for root, dirs, files in os.walk(workdir):
        for file in files:
            if file.endswith(".webm"):
                result.append(VideoTarget(root, file))
    return result