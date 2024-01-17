import argparse
import os
import pickle as pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from load_data import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def inference(model, tokenized_sent, device, cfg):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    batch_size = cfg["params"]["per_device_train_batch_size"]
    dataloader = DataLoader(tokenized_sent, batch_size=batch_size, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                token_type_ids=data["token_type_ids"].to(device),
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def main(cfg):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load tokenizer
    Tokenizer_NAME = cfg["params"]["MODEL_NAME"]
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    ## load my model
    MODEL_NAME = cfg["path"]["MODEL_PATH"]  # model dir.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.parameters
    model.to(device)

    ## load test datset
    test_dataset_dir = cfg["path"]["test_path"]
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    ## predict answer
    pred_answer, output_prob = inference(model, Re_test_dataset, device, cfg)  # model에서 class 추론
    pred_answer = num_to_label(pred_answer, cfg)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    # YAML 파일에서 F1 Score, AUPRC 읽어오기
    with open(MODEL_NAME + "/metrics.yaml", "r") as yaml_file:
        config_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # 값을 읽어오기 - 없으면 -1 반환, 소수점 4자리까지 반올림
    MICRO_F1 = str(round(config_data.get("micro_f1", -1), 4))
    AUPRC = str(round(config_data.get("auprc", -1), 4))

    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )
    save_preds(cfg, MICRO_F1, AUPRC, output)

    #### 필수!! ##############################################
    print("---- Finish! ----")


def save_preds(cfg, micro_f1, auprc, df):
    MODEL_NAME = cfg["params"]["MODEL_NAME"]
    if "/" in MODEL_NAME:
        MODEL_NAME = MODEL_NAME.split("/")[0] + "-" + MODEL_NAME.split("/")[-1]
    DATA_NAME = cfg["path"]["train_path"]
    DATA_NAME = DATA_NAME.split("/")[-1]
    DATA_NAME = DATA_NAME.split(".")[0]
    NUM_EPOCHS = str(cfg["params"]["num_train_epochs"])
    BATCH_SIZE = str(cfg["params"]["per_device_train_batch_size"])
    FILE_NAME = [MODEL_NAME, DATA_NAME, NUM_EPOCHS, BATCH_SIZE, micro_f1, auprc]
    FILE_NAME = "_".join(FILE_NAME) + "_focal_"+ str(cfg["params"]["Get_Focal"]) + ".csv"  #뒤에 focal loss 적용여부 추가
    print(cfg["path"]["submission_path"] + FILE_NAME)
    df.to_csv(cfg["path"]["submission_path"] + FILE_NAME, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="config file path")
    args = parser.parse_args()
    CONFIG_PATH = args.config
    try:
        cfg = load_config(CONFIG_PATH)  # yaml 파일 불러오기
    except:
        cfg = load_config("default_" + CONFIG_PATH)  # config.yaml 파일이 없으면 default 파일 불러오기

    parser.add_argument("--model_dir", type=str, default=cfg["path"]["MODEL_PATH"])

    # model dir
    args = parser.parse_args()
    main(cfg)
