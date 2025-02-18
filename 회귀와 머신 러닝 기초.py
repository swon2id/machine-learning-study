# 회귀(Regression) : 연속적인 수치 값 예측

import numpy as np
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

import matplotlib.pyplot as plt

# 데이터 시각화: 물고기의 길이와 무게 산점도
plt.scatter(perch_length, perch_weight)
plt.xlabel('length(cm)')
plt.ylabel('weight(g)')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor # K-NN 회귀 알고리즘
from sklearn.metrics import mean_absolute_error # 평균 절댓값 오차 측정

# 데이터셋을 훈련 세트와 테스트 세트로 분리 (랜덤 시드: 42)
(train_input, test_input, train_target, test_target) = train_test_split(perch_length, perch_weight, random_state=42)

# 입력 데이터를 2차원 배열로 변환 (모델에 맞는 형태)
train_input = train_input.reshape(-1, 1) # 행은 자동 계산, 열은 1개로 반환
test_input = test_input.reshape(-1, 1)

# k-최근접 이웃 회귀 : 가장 가까운 k개의 이웃 값의 평균으로 예측
# K-최근접 이웃 회귀 알고리즘으로 모델 훈련
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)

# 모델 평가: R^2 결정계수 (1에 가까울수록 좋음)
test_score = knr.score(test_input, test_target)
print(f"테스트 세트 결정계수(R^2) : {test_score}")

# 테스트 세트에 대한 예측 수행
test_prediction = knr.predict(test_input)

# 평균 절댓값 오차 계산 (값이 작을수록 예측이 실제값에 가까움)
mae = mean_absolute_error(test_target, test_prediction)
print(f"평균 절댓값 오차 : {mae}")

'''
과대 적합 vs 과소 적합

과대 적합 : 모델이 훈련 세트에 너무 맞춰져 새로운 데이터에 대한 예측력이 떨어지는 것
- 훈련 세트 점수는 높지만, 테스트 세트 점수가 낮게 나옴

과소 적합 : 모델이 충분히 훈련되지 않아 데이터 패턴을 잘 학습하지 못한 경우
- 훈련 세트와 테스트 세트 점수가 모두 낮은 경우, 테스트 세트 점수가 훈련 세트 보다 높은 경우
'''

# 훈련 세트, 테스트 세트 결정계수 비교
print(
       f"훈련 세트 결정계수(R^2) : {knr.score(train_input, train_target)}",
       f"테스트 세트 결정계수(R^2) : {knr.score(test_input, test_target)}",
       sep="\n"
)

# 모델 개선(튜닝) : 이웃 수 변경
# 과소 적합을 해결하기 위해 모델의 복잡도를 높여야 하고, 복잡도를 높이기 위해서는 k(이웃의 개수)를 줄이면 복잡도 증가
# k 값이 작을 수록 데이터 하나하나에 영향을 많이 받기 때문에, 데이터셋에 대해 세밀하고 민감해져 과적합 가능성을 높일 수 있음

# 이웃의 개수를 3으로 설정
knr.n_neighbors = 3

# 모델 재훈련
knr.fit(train_input, train_target)

# 새로운 훈련 세트 점수
new_train_score = knr.score(train_input, train_target)
print("새로운 훈련 세트 결정계수(R^2):", new_train_score)

# 새로운 테스트 세트 점수
new_test_score = knr.score(test_input, test_target)
print("새로운 테스트 세트 결정계수(R^2):", new_test_score)

# 학습된 모델을 기반으로 예측하기 (길이가 50(cm)인 경우)
print(knr.predict([[50]]))


'''
선형 회귀
- 입력 데이터와 출력 데이터의 선형 관계를 학습 (1차 방정식 형태의 관계를 학습)
- 최소제곱법(Least Squares Method)을 사용하여 잔차(실제값과 예측값의 차이 제곱의 합)를 최소화하는 직선을 찾음
- 또는 2차원 데이터 셋 (x와 y)을 하나의 직선 방정식으로 나타냈을 때 오차가 전체 데이터셋에 대해 가장 작은 직선을 찾음
- 또는 주어진 데이터셋을 가장 잘 표현하는 직선을 찾음
'''
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# 선형 회귀 알고리즘으로 모델 훈련
lr.fit(train_input, train_target)
print(lr.predict([[50]]))

# 훈련 세트의 산점도를 그립니다.
plt.scatter(train_input, train_target)

# 15에서 50까지 1차 방정식 그래프를 그립니다.
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

# 50cm 농어 데이터
plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


'''
다항 회귀
- 독립 변수(입력)와 종속 변수(출력) 사이의 비선형 관계를 모델링하기 위해, 독립 변수의 다항식 항을 추가하여 회귀 분석을 수행하는 알고리즘
- 1차 다항 함수 형태이며, 비선형 관계를 선형 모델로 변환하기 위해 입력 변수에 다항식 변환을 적용
'''

# 길이를 제곱한 항을 훈련 세트에 추가 => 훈련 세트와 테스트 세트 모두 열이 2개로 확장
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
print(train_poly.shape, test_poly.shape)

# train_poly를 사용해 선형 회귀 모델을 다시 훈련
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))

# 훈련한 계수와 절편을 출력
print(lr.coef_, lr.intercept_)
# 출력 예: [  1.01433211 -21.55792498] 116.0502107827827
# - 2차항의 계수, 1차항의 계수, y절편을 의미

# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다.
point = np.arange(15, 50)

# 훈련 세트의 산점도를 그립니다.
plt.scatter(train_input, train_target)

# 15에서 49까지 2차 방정식 그래프를 그립니다.
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)

# 50cm 농어 데이터
plt.scatter(50, 1574, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))