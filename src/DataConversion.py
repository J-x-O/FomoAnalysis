import json
import os

import pandas as pd
from tqdm import tqdm

from src.FrameData import load_combined_frame_data
from src.VideoUtil import VideoTarget


def load_survey_objects(workdir: str) -> dict[str, dict]:
    result = {}
    for root, dirs, files in os.walk(workdir):
        for file in files:
            if file == "survey.json":
                with open(f"{root}/{file}") as f:
                    result[root.split("/")[-1]] = json.load(f)
    return result


def compile_bias(survey_objects: dict[str, dict]) -> pd.DataFrame:
    row_list = []
    for survey_id, survey in survey_objects.items():
        row_list.append({
            "participant": survey_id,
            "demographic_age": survey["demographic"]["age"],
            "demographic_gender": survey["demographic"]["gender"],
            "bias_fear": survey["bias"]["fear"],
            "bias_anger": survey["bias"]["anger"],
            "bias_disgust": survey["bias"]["disgust"],
            "bias_happiness": survey["bias"]["happiness"],
            "bias_sadness": survey["bias"]["sadness"],
            "bias_surprise": survey["bias"]["surprise"],
        })
    return pd.DataFrame(row_list)


def compile_questionnaire(survey_objects: dict[str, dict]) -> pd.DataFrame:
    row_list = []
    for survey_id, survey in survey_objects.items():
        for question in survey["videos"]:
            row_list.append({
                "participant": survey_id,
                "video_id": question["videoID"],
                "actual_category": question["videoID"].split("_")[0].lower().replace("joy", "happiness"),
                "watched_video": question["videoFileName"].split("_")[1],
                "valence": question["valance"],
                "arousal": question["arousal"],
                "intensity": question["intensity"],
                "category": question["category"],
            })
    return pd.DataFrame(row_list)


def percentual_questionnaire(data: pd.DataFrame) -> pd.DataFrame:
    data["valence_percent"] = data["valence"].apply(lambda x: remap(x, 1, 7, 0, 1))
    data["arousal_percent"] = data["arousal"].apply(lambda x: remap(x, 1, 7, 0, 1))
    data["intensity_percent"] = data["intensity"].apply(lambda x: remap(x, 1, 7, 0, 1))
    return data


def remap(value, old_min, old_max, new_min, new_max):
    return ( (value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min#


def normalize_col(df: pd.DataFrame, key: str, from_to: (float, float)) -> pd.DataFrame:
    df[key] = df[key].apply(lambda x: remap(x, from_to[0], from_to[1], 0, 1))
    return df


def retrieve_compiled_video_logs(load_timing: bool = True) -> pd.DataFrame:
    if not os.path.exists("data/compiled_video_logs.csv"):
        df = compile_video_logs()
        df.to_csv("data/compiled_video_logs.csv", index=False)
    else: df = pd.read_csv("data/compiled_video_logs.csv")

    if load_timing: df = load_timing_from_frames(df)
    return df


def load_timing_from_frames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(pd.read_csv("data/frame_rates_average.csv"), on="participant")
    df["time_stamp"] = df["frame"] / df["frame_rate_rounded"]
    return df


def compile_video_logs() -> pd.DataFrame:
    df = pd.read_csv("data/questionnaire.csv")
    final_df = []
    for index, row in tqdm(df.iterrows()):
        target = VideoTarget("data/survey/" + row["participant"], row["video_id"] + ".webm")
        if not target.valid:
            continue
        combined = load_combined_frame_data(target)
        for i, frame in combined.iterrows():
            final_df.append({
                "participant": row["participant"],
                "watched_video": row["watched_video"],
                "actual_category": row["actual_category"],
                "category": row["category"],
                "valence": row["valence"],
                "arousal": row["arousal"],
                "intensity": row["intensity"],
                "model": frame["model"],
                "frame": frame["frame"],
                "axis": frame["axis"],
                "value": frame["value"]
            })
    return pd.DataFrame(final_df)