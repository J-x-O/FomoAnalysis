import json
import os

import pandas as pd


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
                "valance": question["valance"],
                "arousal": question["arousal"],
                "intensity": question["intensity"],
                "category": question["category"],
            })
    return pd.DataFrame(row_list)