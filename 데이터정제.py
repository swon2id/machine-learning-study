import pandas as pd
import numpy as np


df = pd.DataFrame({
  'sex' : [None, 'F', pd.NA, 'M', 'F'],
  'score' : [5, 4, 3, 4, np.nan]
})

print(df)
print(pd.isna(df)) # 각 (x, y) 데이터가 None, pd.NA, np.nan이면 True 아니면 False
print(pd.isna(df).sum()) # 집계 함수
print(df.dropna(subset=['score', 'sex'])) # 결측치 제거

exam = pd.read_csv('exam.csv')화
print(exam)

# 결측치를 평균 값으로 대체하기
random_size = np.random.randint(1, len(exam['math']) + 1)
random_indices = np.random.choice(np.arange(len(exam['math'])), size=random_size, replace=False) # replace로 중복 불허
exam.loc[random_indices, ['math']] = pd.NA

print(exam['math'].sum()) # 결측치가 제외된 상태에서 합산 통계

exam['math'] = exam['math'].fillna(exam['math'].mean().round(0)) # 결측치를 평균 값으로 재할당
print(exam['math'].sum())


# 이상치(아웃라이어)를 결측치로 변환
df2 = pd.DataFrame({
    "sex": [1, 2, 1, 3, 2 ,1],
    "score": [5,4,3,4,3,6]
})

df2['sex'] = np.where(df2['sex'] > 2, np.nan, df2['sex'])
df2['score'] = np.where(df2['score'] > 5, np.nan, df2['score'])
print(df2)
