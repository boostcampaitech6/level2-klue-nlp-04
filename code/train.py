import argparse
import os
import pickle as pickle
import random  # for random seed
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import sklearn
import torch
import yaml
from load_data import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    EarlyStoppingCallback,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import wandb

# for earlystopping, wandb


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        "no_relation",
        "org:top_members/employees",
        "org:members",
        "org:product",
        "per:title",
        "org:alternate_names",
        "per:employee_of",
        "org:place_of_headquarters",
        "per:product",
        "org:number_of_employees/members",
        "per:children",
        "per:place_of_residence",
        "per:alternate_names",
        "per:other_family",
        "per:colleagues",
        "per:origin",
        "per:siblings",
        "per:spouse",
        "org:founded",
        "org:political/religious_affiliation",
        "org:member_of",
        "per:parents",
        "org:dissolved",
        "per:schools_attended",
        "per:date_of_death",
        "per:date_of_birth",
        "per:place_of_birth",
        "per:place_of_death",
        "org:founded_by",
        "per:religion",
    ]
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

    save_preds_to_csv(preds, acc)

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


def label_to_num(label):
    num_label = []
    with open(cfg["path"]["dict_label_to_num"], "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def save_preds_to_csv(preds, acc):
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


# Custom Callback 클래스 정의
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience, early_stopping_threshold, early_stopping_metric, early_stopping_metric_minimize):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_metric_minimize = early_stopping_metric_minimize
        self.best_metric = float("inf") if self.early_stopping_metric_minimize else float("-inf")
        self.waiting_steps = 0

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        current_metric = logs.get(self.early_stopping_metric, None)
        if current_metric is not None:
            if (self.early_stopping_metric_minimize and current_metric < self.best_metric) or (not self.early_stopping_metric_minimize and current_metric > self.best_metric):
                self.best_metric = current_metric
                self.waiting_steps = 0
            else:
                self.waiting_steps += 1

                if self.waiting_steps >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {self.waiting_steps} steps without improvement.")
                    control.should_training_stop = True


def train():
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = cfg["params"]["MODEL_NAME"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    seed = cfg["params"]["seeds"]
    # random seeds setting
    set_seed(seed)  # 랜덤시드 세팅 함수

    # for wandb ,  project="your_project_name", name="your_run_name"
    wandb.init(config=cfg)
    # wandb 에서 이 모델에 어떤 하이퍼 파라미터가 사용되었는지 저장하기 위해, cfg 파일로 설정을 로깅합니다.
    wandb.config.update(cfg)

    # # WandB 콜백 설정 log_model=True 로 하면 최적의 모델이 저장됨.
    # class CustomWandbCallback(TrainerCallback):
    #     def on_log(self, args, state, control, logs=None, model=None, **kwargs):
    #         # WandB에 로그 기록
    #         wandb.log(logs)

    # wandb_callback = CustomWandbCallback()

    # Trainer Callback 생성
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # 조기 중지까지의 기다리는 횟수
        early_stopping_threshold=0.01,  # 개선의 임계값
        early_stopping_metric="eval_loss",  # 평가 지표 (여기서는 eval_loss 사용)
        early_stopping_metric_minimize=True,  # 평가 지표를 최소화해야 하는지 여부
    )

    # load dataset
    train_dataset = load_data(cfg["path"]["train_path"])
    dev_dataset = load_data(cfg["path"]["valid_path"])  # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset["label"].values)
    dev_label = label_to_num(dev_dataset["label"].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device("cuda:0")

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)
    os.environ["WANDB_PROJECT"] = "<Lv2-KLUE>"  # wandb 프로젝트명 설정

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=cfg["path"]["output_dir"],  #                                     output directory
        save_total_limit=cfg["params"]["save_total_limit"],  #                       number of total save model.
        save_steps=cfg["params"]["save_steps"],  #                                   model saving step.
        num_train_epochs=cfg["params"]["num_train_epochs"],  #                       total number of training epochs
        learning_rate=cfg["params"]["learning_rate"],  #                             learning_rate
        per_device_train_batch_size=cfg["params"]["per_device_train_batch_size"],  # batch size per device during training
        per_device_eval_batch_size=cfg["params"]["per_device_eval_batch_size"],  #   batch size for evaluation
        warmup_steps=cfg["params"]["warmup_steps"],  #                               number of warmup steps for learning rate scheduler
        weight_decay=cfg["params"]["weight_decay"],  #                               strength of weight decay
        logging_dir=cfg["path"]["logging_dir"],  #                                   directory for storing logs
        logging_steps=cfg["params"]["logging_steps"],  #                             log saving step.
        evaluation_strategy=cfg["params"]["evaluation_strategy"],  #                 evaluation strategy to adopt during training
        #                                                                           `no`: No evaluation during training.
        #                                                                           `steps`: Evaluate every `eval_steps`.
        #                                                                           `epoch`: Evaluate every end of epoch.
        eval_steps=cfg["params"]["eval_steps"],  #                                   evaluation step.
        load_best_model_at_end=cfg["params"]["load_best_model_at_end"],
        report_to="wandb",  # enable logging to W&B
        run_name=f"{MODEL_NAME}_{cfg['params']['num_train_epochs']:02d}_{cfg['params']['per_device_train_batch_size']}_{cfg['params']['learning_rate']}_{datetime.now(pytz.timezone('Asia/Seoul')):%y%m%d%H%M}",  # name of the W&B run (optional)
    )

    trainer = Trainer(
        model=model,  #                     the instantiated 🤗 Transformers model to be trained
        args=training_args,  #              training arguments, defined above
        train_dataset=RE_train_dataset,  #  training dataset
        eval_dataset=RE_dev_dataset,  #     evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        callbacks=[early_stopping_callback],  # 얼리 스톱핑 콜백과 WandB 콜백 추가
    )

    # train model
    trainer.train()
    model.save_pretrained(cfg["path"]["MODEL_PATH"])

    # evaluate 메서드를 통해 평가 수행
    evaluation_results = trainer.evaluate()

    # evaluation_results에는 compute_metrics 함수에서 반환한 메트릭들이 포함됨
    print("평가결과 : ", evaluation_results)

    # micro f1 score, auprc 추출
    micro_f1 = evaluation_results["eval_micro f1 score"]
    auprc = evaluation_results["eval_auprc"]
    print("micro_f1, auprc : ", micro_f1, auprc)

    # YAML 파일로 저장
    config_data = {"micro_f1": micro_f1, "auprc": auprc}
    with open(cfg["path"]["MODEL_PATH"] + "/metrics.yaml", "w") as yaml_file:
        yaml.dump(config_data, yaml_file)

    wandb.finish()


# yaml 파일 불러오기
def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config


def main():
    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="config file path")
    args = parser.parse_args()
    CONFIG_PATH = args.config

    try:
        cfg = load_config(CONFIG_PATH)  # yaml 파일 불러오기
    except:
        cfg = load_config("default_" + CONFIG_PATH)  # config.yaml 파일이 없으면 default 파일 불러오기

    main()
