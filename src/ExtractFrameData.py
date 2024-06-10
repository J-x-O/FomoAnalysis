import json
import os

from typing import Union

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from networks.DDAMFNpp_affectnet7 import DDAMFNppAffectnet7
from networks.DDAMFNpp_rafdb import DDAMFNppRAFDB
from src.RetinaFaceAlign import transform_stack
from src.VideoUtil import VideoTarget, FrameIterator

class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']

ExtractedEmotion = list[list[float]]
ExtractedEmotionSet = dict[str, ExtractedEmotion]


models = {
    "affectnet7": DDAMFNppAffectnet7(),
    "rafdb": DDAMFNppRAFDB()
}

def has_all_side_cars(video: VideoTarget):
    return video.has_all_side_cars([f"{model_name}.json" for model_name in models.keys()])

def get_missing_models(video: VideoTarget):
    return {k: v for k, v in models.items() if not video.has_side_car(f"{k}.json")}

def get_existing_models(video: VideoTarget):
    return {k: v for k, v in models.items() if video.has_side_car(f"{k}.json")}

def extract_frame_data(video: VideoTarget, skipp_existing: bool = False, console: bool = True) -> ExtractedEmotionSet:
    if skipp_existing and has_all_side_cars(video):
        print(f"Skipping {video.video_title} as all side cars are present.")
        return load_frame_data(video)

    data = { }

    model_selection = models if not skipp_existing else get_missing_models(video)
    for model_name in model_selection.keys():
        data[model_name] = []

    for frame_index, frame in FrameIterator(video, console=console):
        cropped = transform_stack(frame)
        for model_name, model in model_selection.items():
            with torch.no_grad():
                outputs, _, _ = model(cropped)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().tolist()
            data[model_name].append(probabilities[0])

    if not skipp_existing: return data

    for model_name in get_existing_models(video).keys():
        data[model_name] = load_single_frame_data(video, model_name)
    return data

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

def combine_frame_data(data: ExtractedEmotionSet) -> pd.DataFrame:
    row_list = []
    for model_name, model_data in data.items():
        for i, frame in enumerate(model_data):
            for axis_index, axis in enumerate(class_names):
                row_list.append({
                    "model": model_name,
                    "frame": i,
                    "axis": axis,
                    "value": frame[axis_index]})
    return pd.DataFrame(row_list)

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

def paint_frame_data(data: pd.DataFrame):
    sns.lineplot(data=data, x="frame", y="value", hue="axis")
    plt.show()


def clear_cache(video: VideoTarget):
    for appendix in [f"{model_name}.json" for model_name in models.keys()] + ["combined.csv"]:
        side_car = video.side_car(appendix)
        if os.path.exists(side_car):
            os.remove(side_car)

