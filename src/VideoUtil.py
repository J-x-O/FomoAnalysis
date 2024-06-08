import os

import cv2
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
        self.total = count_frames(f"{workdir}/{video_name}")

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
    def __init__(self, video: VideoTarget):
        self.vidcap = cv2.VideoCapture(video.full_path)
        self.frame = None
        self.frame_index = -1
        self.total = video.frame_count
        self.tqdm = tqdm(total=self.total, desc=f"Extracting {video.video_title}")

    def __iter__(self):
        return self

    def __next__(self):
        success, self.frame = self.vidcap.read()
        if success:
            self.frame_index += 1
            self.tqdm.update(1)
            return self.frame_index, self.frame
        self.tqdm.close()
        raise StopIteration


def find_all_videos(workdir: str) -> list[VideoTarget]:
    result = []
    for root, dirs, files in os.walk(workdir):
        for file in files:
            if file.endswith(".webm"):
                result.append(VideoTarget(root, file))
    return result