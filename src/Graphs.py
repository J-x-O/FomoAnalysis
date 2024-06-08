import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def gender_distribution():
    df = pd.read_csv("data/bias.csv")
    data = df["demographic_gender"].value_counts()
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 6))
    plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90)
    plt.show()

def age_distribution():
    df = pd.read_csv("data/bias.csv")
    data = df["demographic_age"].value_counts()
    sns.barplot(x=data.index, y=data.values)
    plt.show()