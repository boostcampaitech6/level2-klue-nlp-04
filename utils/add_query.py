import pandas as pd
from preprocessing import Preprocess

'''
필요한 입력 : train/test 파일 경로

결과물 : sentence가 query + typed entity marker로 변경된 csv파일을 new.csv로 저장해줍니다.

변경된 문장 예시 : 
Before) 
@ * PER * 이순신 @ 은 조선 시대 중기 # ^ POH ^ 무신 #이다

After)
@PER이순신@과 #^POH^무신#의 관계는 무엇인가? [SEP] @ * PER * 이순신 @ 은 조선 시대 중기 # ^ POH ^ 무신#이다 [SEP]
'''

# train.csv 파일 불러오기
file_path = 'train.csv'  # 파일 경로를 실제 파일 경로로 변경해야 합니다.
df = pd.read_csv(file_path)

# Preprocess 클래스 활용
preprocessor = Preprocess(df)

# 전처리 수행
df_processed = preprocessor.add_qa_relation(df)

# entity marker - 필요 시 주석 해제
#df_entity = Preprocess(df_processed).entity_marker(df_processed)
#df_entity.to_csv("new_new.csv")

# typed entity marker
df_typed_entity = preprocessor.typed_entity_marker(df_processed)
df_typed_entity['sentence'] = df_processed['qa_relation'] + ' [SEP] ' + df_typed_entity['sentence'] + '[SEP]'
df_typed_entity.to_csv("new.csv")