import pickle

import numpy as np
import pandas as pd
import sklearn
from load_data import label_keys
from sklearn.metrics import accuracy_score
from train import save_difference


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = label_keys
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


def save_difference(preds, acc, cfg):
    # valid dataset에 대한 predict값과 실제 라벨값을 비교해서 오답파일 생성하는 함수
    difference = pd.read_csv(cfg["path"]["valid_path"])  # 기존 valid_dataset 불러와서 source열 삭제
    difference = difference.drop(columns=["source"])
    with open(cfg["path"]["dict_num_to_label"], "rb") as f:  # 예측한 number형태의 label 값을 label 원 상태로 복구
        dict_num_to_label = pickle.load(f)
    labels = [dict_num_to_label[s] for s in preds]
    difference["predict"] = labels
    condition = difference["predict"] == difference["label"]  # 예측값과 실제값이 같은 것은 위에 정렬하기 위한 코드
    difference_sorted = pd.concat([difference[~condition], difference[condition]])

    MODEL_NAME = cfg["params"]["MODEL_NAME"]  # csv 이름 설정
    if "/" in MODEL_NAME:
        MODEL_NAME = MODEL_NAME.split("/")[0] + "-" + MODEL_NAME.split("/")[-1]

    difference_sorted.to_csv(
        cfg["path"]["difference_path"]
        + "difference_"
        + MODEL_NAME
        + "_"
        + str(cfg["params"]["num_train_epochs"])
        + "_"
        + str(cfg["params"]["per_device_train_batch_size"])
        + "_acc_"
        + str(round(acc, 2))
        + ".csv",
        index=False,
    )
