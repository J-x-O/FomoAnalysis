from tqdm import tqdm

from src.DataConversion import load_survey_objects, compile_bias, compile_questionnaire
from src.Debug import test_img
from src.ExtractFrameData import extract_frame_data, cache_frame_data,  has_all_side_cars
from src.VideoUtil import find_all_videos


def successive_face_analysis():
    videos = find_all_videos("data/survey")
    todo = [v for v in videos if not has_all_side_cars(v)]
    for target in tqdm(todo):
        data = extract_frame_data(target, True)
        cache_frame_data(target, data, True)


def do_stuff():
    survey_data = load_survey_objects("data/survey")
    bias = compile_bias(survey_data)
    bias.to_csv("data/bias.csv", index=False)
    questionnaire = compile_questionnaire(survey_data)
    questionnaire.to_csv("data/questionnaire.csv", index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print("total:", len(find_all_videos("data/survey")))
    test_img()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
