import os
import subprocess
from typing import Union

import cv2
import numpy as np
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm


class VideoTarget:

    def __init__(self, workdir: str, video_name: str):
        self.workdir = workdir
        self.video_name = video_name
        self.video_title = video_name.split(".")[0]
        self.valid = os.path.exists(f"{workdir}/{video_name}")

    def __str__(self):
        return f"{self.workdir} -> {self.video_name}"

    @property
    def frame_count(self):
        return count_frames_path(self.full_path)

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
            if file.endswith(".webm") or file.endswith(".mp4"):
                result.append(VideoTarget(root, file))
    return result


def get_all_frame_rates():
    frame_rates = []
    videos = find_all_videos("data/survey")
    for video in tqdm(videos, desc="Extracting Frame Rates", total=len(videos)):
        frame_rates.append({
            "participant": video.workdir.split("/")[-1].split("\\")[-1],
            "video": video.video_name.split(".")[0],
            "frame_rate": get_frame_rate(video)
        })
    return pd.DataFrame(frame_rates)


def average_frame_rates(df: pd.DataFrame):
    # average the frame rates per participant, and keep both rows
    df = df.groupby("participant").agg({"frame_rate": ["mean"]}).reset_index()
    df["frame_rate_rounded"] = df["frame_rate"].round(0)
    return df


def count_frames_path(path: str) -> int:
    vidcap = cv2.VideoCapture(path)
    count = count_frames(vidcap)
    vidcap.release()
    return count


def count_frames(vidcap: cv2.VideoCapture) -> int:
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        count += 1
    return count


def get_frame_rate(video: VideoTarget):
    # webm is a whore of a format, and you cant get any meta information
    # we need to calculate the expected duration from the reaction video
    df = pd.read_csv("data/reaction_videos_durations.csv")
    title = video.video_title.split("_")
    title = title[1] + "_" + title[2]
    match = df[(df["video"] == title)]
    if len(match) != 1:
        raise Exception("Couldn't find " + video.video_title)
    video_duration = float(match["duration"].values[0])
    reaction_duration = 3 + max(video_duration, 5)
    frame_rate = video.frame_count / reaction_duration
    return frame_rate
