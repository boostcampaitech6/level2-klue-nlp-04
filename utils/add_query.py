import pandas as pd
from preprocessing import Preprocess



'''
필요한 입력 : train/test 파일 경로

결과물 : sentence가 query + typed entity marker로 변경된 csv파일을 new.csv로 저장해줍니다.
'''

# train.csv 파일 불러오기
file_path = 'train.csv'  # 파일 경로를 실제 파일 경로로 변경해야 합니다.
df = pd.read_csv(file_path)


# Preprocess 클래스 활용
preprocessor = Preprocess(df)

# 전처리 수행
df_processed = preprocessor.add_qa_relation(df)

# entity marker
#df_entity = Preprocess(df_processed).entity_marker(df_processed)
#df_entity.to_csv("new_new.csv")

# typed entity marker
df_typed_entity = preprocessor.typed_entity_marker(df_processed)
df_typed_entity['sentence'] = df_processed['qa_relation'] + ' [SEP] ' + df_typed_entity['sentence'] + '[SEP]'
df_typed_entity.to_csv("new.csv")