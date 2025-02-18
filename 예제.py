import numpy as np

# 1번: 1부터 10까지 숫자로 이루어진 1차원 배열 생성 후 모든 요소에 5를 더하기
def q1():
    print(np.arange(1, 11) + 5)

# 2번: 1~9까지 숫자를 사용하여 3x3 크기의 차원 배열 생성 후 출력
def q2():
    print(np.arange(1, 10).reshape(3, 3))

# 3번: 1~20까지 숫자로 이루어진 배열 생성 후 다음을 계산
# 합계, 평균, 최댓 값, 최솟 값
def q3():
    np_arr = np.arange(1, 21)
    print(np_arr.sum())
    print(np_arr.mean())
    print(np_arr.max())
    print(np_arr.min())

# 4번: 0~100까지 난수 10개 생성 후, 50 이상 값 출력
def q4():
    np_arr = np.random.rand(10) * 100
    print(np_arr[np_arr >= 50])

import pandas as pd
import matplotlib.pyplot as plt


midwest = pd.read_csv('midwest.csv')
print(midwest.columns)