# 데이터전처리: 주어진 데이터를 분석에 적합하도록 가공하는 작업
# 데이터 전처리에는 결측치 처리, 중복 제거, 이상치 처리, 데이터 형식 변환, 스케일링 등이 포함됩니다.
# 이를 통해 데이터를 깨끗하고 분석 가능한 형태로 만들어, 분석 및 모델링 성능을 높일 수 있습니다.

import pandas as pd
import numpy as np
from dask.dataframe.methods import assign

# 그룹화 및 집계


# 변수명 수정하기
print("[ 변수명 수정하기 ]")
df_raw = pd.DataFrame({
  'var1' : [1, 2, 1],
  'var2' : [2, 3, 2]
})

df_copy = df_raw.copy() # 데이터를 수정하는 경우, 카피 권장
df_copy.rename(columns={'var1': 'v1', 'var2': 'v2'}, inplace=True) # inplace=True이면 원본에 영향
print(df_copy)
print("*" * 40)


# 파생변수 만들기 (새로운 컬럼 만들기)
exam = pd.read_csv("exam.csv")
print(exam.columns)

mod_exam = exam.assign(
    mean = (exam['math'] + exam['english'] + exam['science']) / 3,
    _pass = np.where(exam['science'] >= 60, 'y', 'n')
)
mod_exam['total'] = exam['math'] + exam['english'] + exam['science']
print(mod_exam)


mpg = pd.read_csv('mpg.csv')
#1번
mpg_copy = mpg.assign(total_연비=mpg['hwy'] + mpg['cty'])
#2번
mpg_copy['avg_연비'] = mpg_copy['total_연비'] / 2
#3번
print(mpg_copy.sort_values(by='avg_연비').head(3))

#4번
def assign_custom(df, **kwargs):
    # DataFrame의 복사본을 만들기
    new_df = df.copy()

    # 각 열을 추가
    for column_name, value in kwargs.items():
        # value가 함수라면 계산된 값을 넣음
        if callable(value):
            new_df[column_name] = value(new_df)
        else:
            new_df[column_name] = value

    return new_df

print(mpg.assign(
    total_연비 = mpg['hwy'] + mpg['cty'],
    avg_연비 = lambda df: df['total_연비'] / 2  # 임시로 total_연비를 참조
).sort_values(by='avg_연비').head(3))


print( (lambda x, y: x + y)(3, 5) ) # 8

def assign_custom(df, **kwargs):
    # DataFrame의 복사본을 만들기
    new_df = df.copy()

    # 각 열을 추가
    for column_name, value in kwargs.items():
        # value가 함수라면 계산된 값을 넣음
        if callable(value):
            new_df[column_name] = value(new_df)
        else:
            new_df[column_name] = value

    return new_df

print(assign_custom(mpg,
    total_연비 = mpg['hwy'] + mpg['cty'],
    avg_연비 = lambda df: df['total_연비'] / 2  # 임시로 total_연비를 참조
).sort_values(by='avg_연비').head(3))

