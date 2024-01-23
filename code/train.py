import argparse
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import torch
import transformers
import yaml
from heatmap import save_difference_png
from load_data import *
from metrics import *
from custom_robertamodel import CustomRobertaForSequenceClassification
from pyprnt import prnt
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import wandb


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def warning_block() -> None:
    # 경고 제거, 함수의 반환은 없습니다.
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore", ".*Some weights of*")
    warnings.filterwarnings("ignore", ".*- This IS*")
    warnings.filterwarnings("ignore", ".*You should probably*")


def save_difference(preds, micro_f1, auprc):
    # valid dataset에 대한 predict값과 실제 라벨값을 비교해서 오답파일 생성하는 함수
    difference = pd.read_csv(cfg["path"]["valid_path"])  # 기존 valid_dataset 불러와서 source열 삭제
    difference = difference.drop(columns=["source"])
    with open(cfg["path"]["dict_num_to_label"], "rb") as f:  # 예측한 number형태의 label 값을 label 원 상태로 복구
        dict_num_to_label = pickle.load(f)
    labels = [dict_num_to_label[s] for s in preds]
    difference["predict"] = labels
    condition = difference["predict"] == difference["label"]  # 예측값과 실제값이 같은 것은 위에 정렬하기 위한 코드
    difference["wrong"] = 1
    difference.loc[condition, "wrong"] = 0  # 틀리면 1, 맞으면 0
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
        + "_f1_"
        + str(round(micro_f1, 2))
        + "_auprc_"
        + str(round(auprc, 2))
        + ".csv",
        index=False,
    )


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
    warning_block()

    seed = cfg["params"]["seeds"]
    set_seed(seed)  # 랜덤시드 세팅 함수

    # load model and tokenizer
    MODEL_NAME = cfg["params"]["MODEL_NAME"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    info = {
        "Model Name": MODEL_NAME,
        "Train Data": cfg["path"]["train_path"].split("/")[-1],
        "Valid Data": cfg["path"]["valid_path"].split("/")[-1],
        "Epoch": cfg["params"]["num_train_epochs"],
        "Learning Rate": cfg["params"]["learning_rate"],
        "Batch Size": cfg["params"]["per_device_train_batch_size"],
    }

    wandb.init(
        config=cfg,
        entity="hello-jobits",
        project="<Lv2-KLUE>",
        name=f"{MODEL_NAME}_{cfg['params']['num_train_epochs']:02f}_{cfg['params']['per_device_train_batch_size']}_{cfg['params']['learning_rate']}_{datetime.now(pytz.timezone('Asia/Seoul')):%y%m%d%H%M}",
    )
    # wandb 에서 이 모델에 어떤 하이퍼 파라미터가 사용되었는지 저장하기 위해, cfg 파일로 설정을 로깅합니다.
    wandb.config.update(cfg)

    # Trainer Callback 생성
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=cfg["params"]["early_stopping_patience"],  # 조기 중지까지의 기다리는 횟수
        early_stopping_threshold=cfg["params"]["early_stopping_threshold"],  # 개선의 임계값
        early_stopping_metric=cfg["params"]["early_stopping_metric"],  # 평가 지표 (여기서는 eval_loss 사용)
        early_stopping_metric_minimize=cfg["params"]["early_stopping_metric_minimize"],  # 평가 지표를 최소화해야 하는지 여부
    )

    # load dataset
    train_dataset = load_data(cfg["path"]["train_path"])
    dev_dataset = load_data(cfg["path"]["valid_path"])

    train_label = label_to_num(train_dataset["label"].values, cfg)
    dev_label = label_to_num(dev_dataset["label"].values, cfg)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device("cuda:0")

    print(device)
    prnt(info)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    # LSTM layer를 추가한 Custom 모델을 사용하려면 아래의 주석 처리된 부분을 사용하시면 됩니다.
    # model = CustomRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model.parameters
    model.to(device)

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
        #                                                                            `no`: No evaluation during training.
        #                                                                            `steps`: Evaluate every `eval_steps`.
        #                                                                            `epoch`: Evaluate every end of epoch.
        eval_steps=cfg["params"]["eval_steps"],  #                                   evaluation step.
        load_best_model_at_end=cfg["params"]["load_best_model_at_end"],
        disable_tqdm=False,
        # save_strategy=cfg["params"]["evaluation_strategy"],
        # metric_for_best_model="micro f1 score",
    )

    get_focal = cfg["params"]["Get_Focal"]
    
    # Focal loss 적용 여부 설정
    if get_focal:
        custom_metrics = compute_metrics_focal
    else:
        custom_metrics = compute_metrics
        
    trainer = Trainer(
            model=model,  #                     the instantiated 🤗 Transformers model to be trained
            args=training_args,  #              training arguments, defined above
            train_dataset=RE_train_dataset,  #  training dataset
            eval_dataset=RE_dev_dataset,  #     evaluation dataset
            compute_metrics=custom_metrics,  #  define metrics function
            callbacks=[early_stopping_callback], # 얼리 스톱핑 콜백과 WandB 콜백 추가
        )

    # train model
    trainer.train()
    model.save_pretrained(cfg["path"]["MODEL_PATH"])

    # evaluate 메서드를 통해 평가 수행
    evaluation_results = trainer.evaluate()
    # evaluation_results에는 compute_metrics 함수에서 반환한 메트릭들이 포함됨

    # micro f1 score, auprc 추출
    micro_f1 = evaluation_results["eval_micro f1 score"]
    auprc = evaluation_results["eval_auprc"]
    acc = evaluation_results["eval_accuracy"]
    results = {"micro_f1": micro_f1, "auprc": auprc, "accuracy": acc}
    prnt(results)

    # difference.csv 파일 출력하기
    pred = trainer.predict(RE_dev_dataset)
    preds = pred.predictions.argmax(-1)
    save_difference(preds, micro_f1, auprc)
    save_difference_png(micro_f1, auprc, cfg)

    # YAML 파일로 저장
    config_data = {"micro_f1": micro_f1, "auprc": auprc}
    with open(cfg["path"]["MODEL_PATH"] + "/metrics.yaml", "w") as yaml_file:
        yaml.dump(config_data, yaml_file)

    wandb.finish()


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
        cfg = load_config("config/default_config.yaml")  # config.yaml 파일이 없으면 default 파일 불러오기

    main()
