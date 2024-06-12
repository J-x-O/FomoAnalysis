import pandas as pd


def filter_from_to_mapping(from_category, to_category):
    df = pd.read_csv("data/questionnaire.csv")
    df = df[(df["category"] == to_category) & (df["actual_category"] == from_category)]
    df = df.drop_duplicates(subset=['watched_video'])
    og = pd.read_csv("data/CowenKeltnerEmotionalVideos.csv")
    df = df.merge(og, left_on="watched_video", right_on="Filename")
    df.to_csv("data/insight/failed_mapping_" + from_category + "_" + to_category + ".csv")


def filter_surprise_happiness_failed_mapping():
    filter_from_to_mapping("surprise", "happiness")


def filter_anger_failed_mapping():
    filter_from_to_mapping("anger", "surprise")
    filter_from_to_mapping("anger", "sadness")