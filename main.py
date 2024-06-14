import pandas as pd

from src.DataConversion import load_survey_objects, compile_bias, compile_questionnaire, retrieve_compiled_video_logs, \
    compile_video_logs
from src.DataQuickAccess import filter_surprise_happiness_failed_mapping, filter_anger_failed_mapping
from src.Debug import average_age, shortest_video
from src.Graphs import *
from src.VideoUtil import cache_all_frame_rates


def do_stuff():
    survey_data = load_survey_objects("data/survey")

    bias = compile_bias(survey_data)
    bias.to_csv("data/bias.csv", index=False)

    questionnaire = compile_questionnaire(survey_data)
    questionnaire.to_csv("data/questionnaire.csv", index=False)

    df = compile_video_logs()
    df.to_csv("data/compiled_video_logs.csv", index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plot_all()
    # print(shortest_video())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
