# Baseline 코드 리팩토링

## 변경된 사항

1. Makefile 추가로 사용성 증진 예상 `~~run.py` 만들기 귀찮아서 그런거 아님~~
2. 커밋을 하면 자동으로 [코드 포맷팅](https://www.notion.so/7664352c799249dfa17e7558e6aa2eb7?pvs=21)
3. 폴더 분리
4. gitignore 수정으로 기본 세팅의 config 파일을 포함


## 초기 설정(중요!)

1. `default_config.yaml` 파일을 수정하지 않고, **복사**하여 `config.yaml` 생성하기
    1. 이름 안맞추면 default로 실행
    2. `default_config.yaml` 파일은 커밋 대상에 올라가므로 수정하면 커밋에 포함됨
2. `dataset/train`, `dataset/test`에 데이터 파일 넣기


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
