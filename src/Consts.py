from typing import Union

import seaborn as sns
from pandas import Series

emotion_classes_network = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']
emotion_classes_vote = ['neutral', 'happiness', 'sadness', 'surprise', 'fear', 'disgust', 'anger']
emotion_classes_og_dataset = ['happiness', 'sadness', 'surprise', 'fear', 'disgust', 'anger']

palette_full = sns.color_palette(['#8C8C8C', '#CCB974', '#64B5CD', '#DD8452', '#55A868', '#8172B3', '#C44E52'])
palette_no_neutral = sns.color_palette(['#CCB974', '#64B5CD', '#DD8452', '#55A868', '#8172B3', '#C44E52'])

def network_to_vote(emotion: Union[str, Series]) -> str:
    if isinstance(emotion, Series):
        return emotion.apply(lambda x: emotion_classes_vote[emotion_classes_network.index(x)])
    return emotion_classes_vote[emotion_classes_network.index(emotion)]