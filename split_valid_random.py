"""랜덤하게 8:2 비율로 train_new.csv, valid.csv 파일 분할
    
Description:
   1. config.yaml을 통해 train.csv 경로 설정
   2. train.csv와 같은 위치에 분할된 데이터셋 train_new.csv, valid.csv가 생성됨 
   3. 호출 : split_valid(train_path)
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

# 1. 랜덤하게 8:2 비율로 train_new.csv, valid.csv 파일 쪼개기
def split_valid(train_path):
    
    # train.csv 파일을 읽어오기
    df_train = pd.read_csv(cfg["path"]["train_path"])  # 'train.csv'의 경로를 입력해야 합니다.

    # train 데이터를 8:2 비율로 train과 valid로 분할
    df_train, df_valid = train_test_split(df_train, test_size=0.2, random_state=42)

    # train_new.csv, valid.csv로 분할된 데이터를 저장
    df_train.to_csv('/data/ephemeral/dataset/train/train_new.csv', index=False)
    df_valid.to_csv(cfg["path"]["valid_path"], index=False)
    

if __name__ == '__main__':
    cfg = load_config("config.yaml") # yaml 파일 불러오기
    train_path = cfg["path"]["valid_path"]
    split_valid()