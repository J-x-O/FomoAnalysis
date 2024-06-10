import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.axes import Axes
from seaborn import FacetGrid

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
    melted_df = df.melt(id_vars=["actual_category", "watched_video"], value_vars=[key, "true_" + key, key + "_diff"], var_name="col", value_name="value")
    melted_df = melted_df[~((melted_df['col'] == 'true_' + key) & melted_df.duplicated(['col', 'watched_video']))]
    g = sns.FacetGrid(melted_df, col="col", hue="actual_category", sharex=False, sharey=False, aspect=1.2)
    g.map(sns.kdeplot, "value", bw_adjust=1)
    g.add_legend()
    g.axes[0, 2].axvline(x=0, color='black', linestyle='--')
    return g

def plot_valence_distribution(): return plot_distribution("valence")
def plot_arousal_distribution(): return plot_distribution("arousal")
def plot_intensity_distribution(): return plot_distribution("intensity")
def plot_distribution(key: str) -> Axes:
    df = pd.read_csv("data/questionnaire.csv")
    return sns.kdeplot(df, x=key, hue="actual_category")

def plot_true_valence_distribution(): return plot_true_distribution("valence")
def plot_true_arousal_distribution(): return plot_true_distribution("arousal")
def plot_true_distribution(key: str) -> Axes:
    df = pd.read_csv("data/questionnaire.csv")
    df = load_single_video_data(df, key, normalize=True)
    return sns.kdeplot(df, x="true_" + key, hue="actual_category")


def plot_valence_matching(): return plot_matching("valence")
def plot_arousal_matching(): return plot_matching("arousal")
def plot_matching(key: str) -> Axes:
    df = pd.read_csv("data/questionnaire.csv")
    df = normalize_col(df, key, (1, 7))
    df = load_single_video_data(df, key, normalize=True)
    df[key + "_diff"] = df[key] - df["true_" + key]
    ax = sns.kdeplot(df, x=key + "_diff", hue="actual_category")
    ax.axvline(x=0, color='black', linestyle='--')
    return ax

def plot_video_watch_count() -> Axes:
    df = pd.read_csv("data/questionnaire.csv")
    data = df["watched_video"].value_counts().rename_axis('watched_video').to_frame('index')
    data = data.merge(df[["watched_video", "actual_category"]], on="watched_video")
    data = data.sort_values("actual_category", ascending=False)
    fig, ax = pyplot.subplots(figsize=(16.0, 7.0))
    ax = sns.barplot(data, x="watched_video", y="index", hue="actual_category", ax=ax)
    plt.xticks(rotation=45, ha='right')
    return ax


def plot_pre_reaction() -> Axes:
    df = retrieve_compiled_video_logs()
    df = df[df["time_stamp"] < 3]
    # find out the dominant emotion for each frame, of every video from every participant
    # df = df.loc[df.groupby(["participant", "watched_video", "frame"])['value'].idxmax()]
    return sns.lmplot(data=df, x="time_stamp", y="value", col="axis", hue="model", scatter_kws={"s": 20, "alpha": 0.1, "edgecolors": 'none'})

def plot_all_reaction_fill() -> FacetGrid:
    df = retrieve_compiled_video_logs()
    df = df[df["time_stamp"] < 8]
    g = sns.displot(data=df, x="time_stamp", weights="value", col="model", hue="axis", kind="kde", multiple="fill", clip=(0.0, 8.0))
    return g

def plot_all_reaction_fill_by_emotion() -> FacetGrid:
    df = retrieve_compiled_video_logs()
    df = df[df["time_stamp"] < 8]
    g = sns.displot(data=df, x="time_stamp", weights="value", row="model", col="category", hue="axis", kind="kde", multiple="fill", clip=(0.0, 8.0))
    return g


def plot_all_reaction_fill_by_original_category() -> FacetGrid:
    df = retrieve_compiled_video_logs()
    df = df[df["time_stamp"] < 8]
    g = sns.displot(data=df, x="time_stamp", weights="value", row="model", col="actual_category", hue="axis", kind="kde", multiple="fill", clip=(0.0, 8.0))
    return g



all_plots = [# plot_gender_distribution, plot_age_distribution,
             # plot_confusion_matrix, plot_video_watch_count,
             # plot_valence_comparison, plot_arousal_comparison, plot_intensity_distribution,
             # plot_pre_reaction,
             # plot_all_reaction_fill
             plot_all_reaction_fill_by_original_category
]