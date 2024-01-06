"""클래스 분포가 같도록 8:2 비율로 train_new.csv, valid.csv 파일 분할
    
Description:
   1. config.yaml을 통해 train.csv 경로 설정
   2. train.csv와 같은 위치에 분할된 데이터셋 train_new.csv, valid.csv가 생성됨 
   3. 호출 : split_stratify_valid(cfg["path"]["train_path"], cfg["path"]["valid_path"])
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold


# yaml 파일 불러오기
def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

# 2. 클래스 분포 비율 맞춰서 - 8:2로 train_new, valid.csv 파일 쪼개기
def split_stratify_valid(train_path, valid_path):
    # train.csv 파일을 읽어오기
    df_train = pd.read_csv(train_path)  # 'train.csv'의 경로를 입력해야 합니다.

    # 각 클래스 별로 데이터를 나누기
    unique_labels = df_train['label'].unique()

    dfs_train = []
    dfs_valid = []

    for label in unique_labels:
        df_label = df_train[df_train['label'] == label]

        # label 클래스에 해당하는 데이터를 8:2 비율로 train과 valid로 분할
        df_train_label, df_valid_label = train_test_split(df_label, test_size=0.2, random_state=42)

        dfs_train.append(df_train_label)
        dfs_valid.append(df_valid_label)

    # 각 클래스 별로 나뉜 train 데이터와 valid 데이터를 하나로 합치기
    df_train = pd.concat(dfs_train)
    df_valid = pd.concat(dfs_valid)

    # 클래스 분포 확인
    print("Train 클래스 분포:")
    print(df_train['label'].value_counts(normalize=True))
    print("\nValid 클래스 분포:")
    print(df_valid['label'].value_counts(normalize=True))

    # train.csv, valid.csv로 분할된 데이터를 저장
    df_train.to_csv('/data/ephemeral/dataset/train/train_new.csv', index=False)
    df_valid.to_csv(valid_path, index=False)

if __name__ == '__main__':
    cfg = load_config("config.yaml") # yaml 파일 불러오기
    split_stratify_valid(cfg["path"]["train_path"], cfg["path"]["valid_path"])
   