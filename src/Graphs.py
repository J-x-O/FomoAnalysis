import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.axes import Axes
from seaborn import FacetGrid

from src.Consts import emotion_classes_network, emotion_classes_vote, emotion_classes_og_dataset, network_to_vote, \
    palette_no_neutral, palette_full
from src.DataConversion import remap, normalize_col, retrieve_compiled_video_logs, load_timing_from_frames


def plot_all():
    for plot in all_plots:
        plt.figure()
        plot()
        plt.savefig(f"plots/{plot.__name__}.png")


def plot_gender_distribution() -> Axes:
    df = pd.read_csv("data/bias.csv")
    data = df["demographic_gender"].value_counts()
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 6))
    plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90)
    return plt.gca()


def plot_age_distribution() -> Axes:
    df = pd.read_csv("data/bias.csv")
    data = df["demographic_age"].value_counts()
    for i in range(min(data.index), max(data.index) + 1):
        if i not in data:
            data[i] = 0
    return sns.barplot(x=data.index, y=data.values)


def plot_confusion_matrix() -> Axes:
    df = pd.read_csv("data/questionnaire.csv")
    conf_matrix = pd.crosstab(df['actual_category'], df['category'])
    return sns.heatmap(conf_matrix, annot=True)


def load_single_video_data(df: pd.DataFrame, key: str, normalize: bool = False) -> pd.DataFrame:
    other = pd.read_csv("data/CowenKeltnerEmotionalVideos.csv") \
        .filter(["Filename", key]) \
        .rename(columns={key: "true_" + key, "Filename": "watched_video"})
    if normalize:
        other = normalize_col(other, "true_" + key, (1, 9))
    return df.merge(other, on="watched_video")


def plot_valence_comparison(): return plot_triple_comparison("valence")
def plot_arousal_comparison(): return plot_triple_comparison("arousal")
def plot_triple_comparison(key: str) -> FacetGrid:
    df = pd.read_csv("data/questionnaire.csv")
    df = normalize_col(df, key, (1, 7))
    df = load_single_video_data(df, key, normalize=True)
    df[key + "_diff"] = df[key] - df["true_" + key]
    melted_df = df.melt(id_vars=["actual_category", "watched_video"], value_vars=[key, "true_" + key, key + "_diff"],
                        var_name="col", value_name="value")
    melted_df = melted_df[~((melted_df['col'] == 'true_' + key) & melted_df.duplicated(['col', 'watched_video']))]
    sns.set_palette(palette_no_neutral)
    g = sns.FacetGrid(melted_df, col="col", hue="actual_category", hue_order=emotion_classes_og_dataset, sharex=False, sharey=False, aspect=1.2)
    g.map(sns.kdeplot, "value", bw_adjust=1)
    g.add_legend()
    g.axes[0, 2].axvline(x=0, color='black', linestyle='--')
    return g


def plot_valence_distribution(): return plot_distribution("valence")
def plot_arousal_distribution(): return plot_distribution("arousal")
def plot_intensity_distribution(): return plot_distribution("intensity")
def plot_distribution(key: str) -> Axes:
    df = pd.read_csv("data/questionnaire.csv")
    sns.set_palette(palette_no_neutral)
    return sns.kdeplot(df, x=key, hue="actual_category", hue_order=emotion_classes_og_dataset)


def plot_true_valence_distribution(): return plot_true_distribution("valence")
def plot_true_arousal_distribution(): return plot_true_distribution("arousal")
def plot_true_distribution(key: str) -> Axes:
    df = pd.read_csv("data/questionnaire.csv")
    df = load_single_video_data(df, key, normalize=True)
    sns.set_palette(palette_no_neutral)
    return sns.kdeplot(df, x="true_" + key, hue="actual_category", hue_order=emotion_classes_og_dataset)


def plot_valence_matching(): return plot_matching("valence")
def plot_arousal_matching(): return plot_matching("arousal")
def plot_matching(key: str) -> Axes:
    df = pd.read_csv("data/questionnaire.csv")
    df = normalize_col(df, key, (1, 7))
    df = load_single_video_data(df, key, normalize=True)
    df[key + "_diff"] = df[key] - df["true_" + key]
    sns.set_palette(palette_no_neutral)
    ax = sns.kdeplot(df, x=key + "_diff", hue="actual_category", hue_order=emotion_classes_og_dataset)
    ax.axvline(x=0, color='black', linestyle='--')
    return ax


def plot_relation_bias_valence() -> FacetGrid: return plot_relation_bias_key("valence")
def plot_relation_bias_arousal() -> FacetGrid: return plot_relation_bias_key("arousal")
def plot_relation_bias_intensity() -> FacetGrid: return plot_relation_bias_key("intensity")
def plot_relation_bias_key(key: str) -> FacetGrid:
    df = pd.read_csv("data/questionnaire.csv")
    df = df[df["category"] != "neutral"]
    df = df.merge(pd.read_csv("data/bias.csv"), on="participant")
    df["bias"] = df.apply(lambda x: x["bias_" + x["category"]], axis=1)
    sns.set_palette(palette_no_neutral)
    g = sns.lmplot(data=df, x="bias", y=key, col="category", col_order=emotion_classes_og_dataset,
                   hue="category", hue_order=emotion_classes_og_dataset)
    return g


def order_by_emotion_class_og_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df["actual_category_index"] = df["actual_category"].apply(lambda x: emotion_classes_og_dataset.index(x))
    return df.sort_values(["actual_category_index", "watched_video"])

def plot_video_watch_count() -> Axes:
    df = pd.read_csv("data/questionnaire.csv")
    data = df["watched_video"].value_counts().rename_axis('watched_video').to_frame('index')
    data = data.merge(df[["watched_video", "actual_category"]], on="watched_video")
    data = order_by_emotion_class_og_dataset(data)
    fig, ax = pyplot.subplots(figsize=(16.0, 7.0))
    sns.set_palette(palette_no_neutral)
    ax = sns.barplot(data, x="watched_video", y="index", hue="actual_category", ax=ax, hue_order=emotion_classes_og_dataset)
    plt.xticks(rotation=45, ha='right')
    return ax


def plot_pre_reaction() -> Axes:
    df = retrieve_compiled_video_logs()
    df = df[df["time_stamp"] < 3]
    # find out the dominant emotion for each frame, of every video from every participant
    # df = df.loc[df.groupby(["participant", "watched_video", "frame"])['value'].idxmax()]
    sns.set_palette(sns.color_palette("deep"))
    return sns.lmplot(data=df, x="time_stamp", y="value", col="axis", hue="model",
                      scatter_kws={"s": 20, "alpha": 0.1, "edgecolors": 'none'}, col_order=emotion_classes_network)


def plot_all_reaction_fill() -> FacetGrid:
    df = retrieve_compiled_video_logs()
    df = df[df["time_stamp"] < 8]
    sns.set_palette(palette_full)
    g = sns.displot(data=df, x="time_stamp", weights="value", col="model", hue="axis", kind="kde", multiple="fill",
                    clip=(0.0, 8.0), hue_order=emotion_classes_network)
    return g


def plot_all_reaction_fill_by_emotion() -> FacetGrid:
    df = retrieve_compiled_video_logs()
    df = df[df["time_stamp"] < 8]
    sns.set_palette(palette_full)
    g = sns.displot(data=df, x="time_stamp", weights="value", row="model", col="category", hue="axis", kind="kde",
                    multiple="fill", clip=(0.0, 8.0), hue_order=emotion_classes_network, col_order=emotion_classes_vote)
    return g


def plot_all_reaction_fill_by_original_category() -> FacetGrid:
    df = retrieve_compiled_video_logs()
    df = df[df["time_stamp"] < 8]
    sns.set_palette(palette_full)
    g = sns.displot(data=df, x="time_stamp", weights="value", row="model", col="actual_category", hue="axis",
                    kind="kde", multiple="fill", clip=(0.0, 8.0), hue_order=emotion_classes_network,
                    col_order=emotion_classes_og_dataset)
    return g


def plot_most_confident_guess() -> FacetGrid:
    df = retrieve_compiled_video_logs()
    df = df[df["time_stamp"] < 8]
    df = df.loc[df.groupby(["participant", "watched_video", "frame"])['value'].idxmax()]
    sns.set_palette(palette_full)
    g = sns.displot(data=df, x="time_stamp", weights="value", row="model", col="category", hue="axis", kind="kde",
                    multiple="fill", clip=(0.0, 8.0), hue_order=emotion_classes_network, col_order=emotion_classes_vote)
    return g


def load_correct_guess_data() -> pd.DataFrame:
    df = retrieve_compiled_video_logs()
    df = df.loc[df.groupby(["participant", "watched_video", "frame", "model"])['value'].idxmax()]
    df["is_correct"] = df["category"] == network_to_vote(df["axis"])
    return df


def plot_correct_guess() -> FacetGrid:
    df = load_correct_guess_data()
    sns.set_palette(sns.color_palette("deep"))
    g = sns.displot(data=df, x="time_stamp", row="model", col="category", hue="is_correct", kind="kde",
                    multiple="fill", clip=(0.0, 8.0), col_order=emotion_classes_vote)
    return g


def load_averaged_correct_guess_data() -> pd.DataFrame:
    df = load_correct_guess_data()
    df = df[3 < df["time_stamp"]]
    df_grouped = df.groupby(["participant", "watched_video", "model"])["is_correct"].mean().reset_index()
    video_info = df[["participant", "watched_video", "actual_category", "category", "valence", "arousal", "intensity"]].drop_duplicates()
    return df_grouped.merge(video_info, on=["participant", "watched_video"])

def plot_correct_guess_highest() -> Axes:
    df = load_averaged_correct_guess_data()
    df = order_by_emotion_class_og_dataset(df)
    fig, ax = pyplot.subplots(figsize=(16.0, 7.0))
    sns.set_palette(palette_no_neutral)
    ax = sns.barplot(data=df, x="watched_video", y="is_correct", hue="actual_category", ax=ax, hue_order=emotion_classes_og_dataset)
    plt.xticks(rotation=45, ha='right')
    return ax


def plot_correct_guess_relation_intensity() -> Axes: return plot_correct_guess_relation_by("intensity")
def plot_correct_guess_relation_valence() -> Axes: return plot_correct_guess_relation_by("valence")
def plot_correct_guess_relation_arousal() -> Axes: return plot_correct_guess_relation_by("arousal")
def plot_correct_guess_relation_by(key: str) -> Axes:
    df = load_averaged_correct_guess_data()
    sns.set_palette(palette_full)
    ax = sns.lmplot(data=df, x=key, y="is_correct", row="model", col="category", col_order=emotion_classes_vote,
                    hue="category", hue_order=emotion_classes_vote)
    return ax


def plot_correct_guess_relation_bias() -> Axes:
    df = load_averaged_correct_guess_data()
    df = df[df["category"] != "neutral"]
    df = df.merge(pd.read_csv("data/bias.csv"), on="participant")
    df["bias"] = df.apply(lambda x: x["bias_" + x["category"]], axis=1)
    classes = [x for x in emotion_classes_vote if x != "neutral"]
    sns.set_palette(palette_no_neutral)
    ax = sns.lmplot(data=df, x="bias", y="is_correct", row="model", col="category", col_order=classes,
                    hue="category", hue_order=classes)
    return ax


def plot_correct_guess_relation_age() -> Axes:
    df = load_averaged_correct_guess_data()
    df = df.merge(pd.read_csv("data/bias.csv"), on="participant")
    sns.set_palette(sns.color_palette("deep"))
    ax = sns.lmplot(data=df, x="demographic_age", y="is_correct")
    return ax

def plot_correct_guess_relation_gender() -> Axes:
    df = load_averaged_correct_guess_data()
    df = df.merge(pd.read_csv("data/bias.csv"), on="participant")
    sns.set_palette(palette_full)
    ax = sns.catplot(data=df, x="demographic_gender", y="is_correct", hue="category", hue_order=emotion_classes_vote, col="model", kind="bar")
    return ax



# todo: bias, gender and age for both the facial expression and the actually felt emotion

all_plots = [
    # plot_gender_distribution, plot_age_distribution,
    # plot_confusion_matrix,
    # plot_valence_comparison, plot_arousal_comparison, plot_intensity_distribution,
    plot_relation_bias_valence, plot_relation_bias_arousal, plot_relation_bias_intensity,
    # plot_video_watch_count,
    # plot_pre_reaction,
    # plot_all_reaction_fill,
    # plot_all_reaction_fill_by_emotion,
    # plot_all_reaction_fill_by_original_category,
    # plot_most_confident_guess,
    # plot_correct_guess,
    # plot_correct_guess_highest,
    # plot_correct_guess_relation_intensity, plot_correct_guess_relation_valence, plot_correct_guess_relation_arousal,
    # plot_correct_guess_relation_bias,
    # plot_correct_guess_relation_age, plot_correct_guess_relation_gender,
]
