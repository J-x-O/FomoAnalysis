from src.DataConversion import load_survey_objects, compile_bias, compile_questionnaire, retrieve_compiled_video_logs
from src.Graphs import *


def do_stuff():
    survey_data = load_survey_objects("data/survey")
    bias = compile_bias(survey_data)
    bias.to_csv("data/bias.csv", index=False)
    questionnaire = compile_questionnaire(survey_data)
    questionnaire.to_csv("data/questionnaire.csv", index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plot_all()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
