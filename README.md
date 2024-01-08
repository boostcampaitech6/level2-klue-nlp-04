# Baseline 코드 리팩토링 + 거기에 얼리스탑핑 적용!

## 변경된 사항
1. wandb 연결 및 wandb 기능을 이용하여 얼리스탑핑을 적용하였습니다.
2. 

### 나머진 기존 베이스라인 코드랑 같아요 헤헤

## 사용방법

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
