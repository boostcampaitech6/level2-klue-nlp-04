# [FEAT] Split Data 9:1 Test + Focal Loss test

## 변경된 사항
1. utils/add_query.py 생성


* sentence의 앞에 질문 형태의 쿼리를 추가합니다.

* 추가할 때, Type Entity Marker의 형식에 맞게 추가해줍니다.

* add_query.py는 파일 경로를 config.yaml과 연동하지 않았습니다. 직접 설정후 add_query.py를 실행해주세요.



## 초기 설정(중요!) - 동일

1. `default_config.yaml` 파일을 수정하지 않고, **복사**하여 `config.yaml` 생성하기
    1. 이름 안맞추면 default로 실행
    2. `default_config.yaml` 파일은 커밋 대상에 올라가므로 수정하면 커밋에 포함됨
    3. `default_config.yaml`이 변경 되었으니 다시 확인 요망!
2. `dataset/train`, `dataset/test`에 데이터 파일 넣기
   1. `dataset/train`에 `valid.csv` 파일을 추가하기
   2. `code/split_valid_*` 파일을 사용하면 자동으로 `valid.csv` 생성이 가능!
   3. 그러면 validation을 제외한 데이터는 `train_new.csv` 파일로 생성
   4. 그리고 `config.yaml` 파일에서 train_path를 `train_new.csv` 으로 변경!
4. `config2.yaml`처럼 config 디렉토리에 `config`뒤에 이름을 자유롭게 .yaml 파일을 추가해주세요(개수 제한 없음)

## 사용방법(동일)

### 1. train + inference

```bash
make run
```

1-1. train only

```bash
make train
```

1-2. inference only

```bash
make inference
```

## 2. 기존 학습 모델 제거

```bash
make clean
```

## 3. 연속으로 학습시키키
```bash
make all
```

만약 실행이 안된다면 read_config.sh 파일의 실행권한이 없는 것이므로 아래와 같이 실행 권한을 부여하면 됩니다.
```bash
chmod 755 ./read_config.sh
```
