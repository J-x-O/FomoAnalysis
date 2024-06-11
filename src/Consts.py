from typing import Union

from pandas import Series

emotion_classes_network = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']
emotion_classes_vote = ['neutral', 'happiness', 'sadness', 'surprise', 'fear', 'disgust', 'anger']
emotion_classes_og_dataset = ['happiness', 'sadness', 'surprise', 'fear', 'disgust', 'anger']


def network_to_vote(emotion: Union[str, Series]) -> str:
    if isinstance(emotion, Series):
        return emotion.apply(lambda x: emotion_classes_vote[emotion_classes_network.index(x)])
    return emotion_classes_vote[emotion_classes_network.index(emotion)]