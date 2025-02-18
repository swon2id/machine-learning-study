import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42)

'''
다항 특성 변환(Polynomial Features)은 기존의 입력 특성을 제곱, 세제곱 등 고차항으로 확장하는 과정입니다.
예를 들어, 입력 특성 x가 있으면 x^2, x^3 등을 추가하여 비선형적 관계도 학습할 수 있게 합니다.
이는 선형 회귀 모델로도 비선형 데이터를 잘 예측할 수 있게 합니다.
'''
poly = PolynomialFeatures(degree=5, include_bias=False) # 다항식 차수: 5, 절편 제거
train_poly = poly.fit_transform(train_input)
test_poly = poly.transform(test_input)

# 선형 회귀 알고리즘을 사용하여 훈련
lr = LinearRegression()
lr.fit(train_poly, train_target)

# 훈련 세트 및 테스트 세트 점수 (결정 계수 출력)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))