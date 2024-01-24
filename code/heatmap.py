import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from load_data import label_keys, label_to_num
from matplotlib.colors import LogNorm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def save_difference_png(micro_f1, auprc, cfg):
    model_name = cfg["params"]["MODEL_NAME"]  # csv 이름 설정
    if "/" in model_name:
        model_name = model_name.split("/")[0] + "-" + model_name.split("/")[-1]

    file_name = (
        cfg["path"]["difference_path"]
        + "difference_"
        + model_name
        + "_"
        + str(cfg["params"]["num_train_epochs"])
        + "_"
        + str(cfg["params"]["per_device_train_batch_size"])
        + "_f1_"
        + str(round(micro_f1, 2))
        + "_auprc_"
        + str(round(auprc, 2))
    )

    y_true, y_pred = read_csv(file_name)
    y_true = label_to_num(y_true)
    y_pred = label_to_num(y_pred)

    plot_confusion_matrix(y_true, y_pred, file_name)
    plot_confusion_matrix_norm(y_true, y_pred, file_name)
    calculate_metrics(y_true, y_pred)


def read_csv(file_name):
    df = pd.read_csv(file_name + ".csv")
    y_true = df["label"].tolist()
    y_pred = df["predict"].tolist()

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, file_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_keys, yticklabels=label_keys)
    plt.title(file_name)
    plt.yticks(rotation=0)
    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    plt.savefig(f"{file_name}.png", dpi=300)
    plt.close()


def plot_confusion_matrix_norm(y_true, y_pred, file_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_keys, yticklabels=label_keys, norm=LogNorm())
    plt.title(file_name)
    plt.yticks(rotation=0)
    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    plt.savefig(f"{file_name}_norm.png", dpi=300)
    plt.close()


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
