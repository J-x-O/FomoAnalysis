import json
import os
from typing import Union

import pandas as pd

from src.Consts import emotion_classes
from src.VideoUtil import VideoTarget


ExtractedEmotion = list[list[float]]
ExtractedEmotionSet = dict[str, ExtractedEmotion]


models = [ "affectnet7", "rafdb" ]


def has_all_side_cars(video: VideoTarget):
    return video.has_all_side_cars([f"{model_name}.json" for model_name in models.keys()])


def load_frame_data(video: VideoTarget) -> ExtractedEmotionSet:
    loaded = {}
    for model_name in models.keys():
        loaded[model_name] = load_single_frame_data(video, model_name)
    return loaded

def load_single_frame_data(video: VideoTarget, model_name: str) -> ExtractedEmotion:
    with open(video.side_car(f"{model_name}.json"), "r") as f:
        return json.loads(f.read())

def load_combined_frame_data(video: VideoTarget) -> pd.DataFrame:
    if video.has_side_car("combined.csv"):
        return pd.read_csv(video.side_car("combined.csv"))
    if has_all_side_cars(video):
        data = load_frame_data(video)
        return combine_frame_data(data)
    raise FileNotFoundError(f"Combined frame data not found for {video.full_path}")


def combine_frame_data(data: ExtractedEmotionSet) -> pd.DataFrame:
    row_list = []
    for model_name, model_data in data.items():
        for i, frame in enumerate(model_data):
            for axis_index, axis in enumerate(emotion_classes):
                row_list.append({
                    "model": model_name,
                    "frame": i,
                    "axis": axis,
                    "value": frame[axis_index]})
    return pd.DataFrame(row_list)

def cache_frame_data(video: VideoTarget, data: Union[ExtractedEmotionSet, pd.DataFrame], skipp_existing: bool = False):
    if isinstance(data, dict):
        for model_name, model_data in data.items():
            if not skipp_existing or not video.has_side_car(f"{model_name}.json"):
                with open(video.side_car(f"{model_name}.json"), "w+") as f:
                    f.write(json.dumps(model_data))
    if isinstance(data, pd.DataFrame):
        if not skipp_existing or not video.has_side_car("combined.csv"):
            combined = combine_frame_data(data)
            combined.to_csv(video.side_car("combined.csv"), index=False)


def clear_cache(video: VideoTarget):
    for appendix in [f"{model_name}.json" for model_name in models.keys()] + ["combined.csv"]:
        side_car = video.side_car(appendix)
        if os.path.exists(side_car):
            os.remove(side_car)
