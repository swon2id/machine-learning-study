"""
넘파이(numpy)는 파이썬에서 과학적인 계산을 위한 핵심 라이브러리
빠른 수치 연산과 다차원 배열 연산에 특화되어 있다.

주의점은 개별 요소는 동일한 데이터 타입이어야 한다.
"""

import numpy as np

# 1차원 numpy 배열 생성
a1 = np.array([0, 1, 2, 3, 4, 5]) # 리스트를 nupmy 배열로 변환
print(a1)
print(a1.shape) # 행과 열 (6,)
print(a1.dtype) # int32
print("-"*20)

# 2차원 numpy 배열 생성
a2 = np.array(([1,2,3], [4,5,6])) # 다차원 배열은 튜플로 만들어서 전달한다
print(a2)
print(a2.shape) # 행과 열 (2, 3)
print(a2.dtype) # int32
print("-"*20)

# range 기반 numpy 배열 생성
a3 = np.arange(0, 10, 2) # [start end) step마다 할당
print(a3)
print("-"*20)

a4 = np.arange(10) # [0 end) step=1
print(a4)
print("-"*20)

# 차원 재배열
a5 = np.arange(12).reshape(4, 3) # 설정 할 행과 열에 대해, 요소 수가 일치해야 함
print(a5)
print("-"*20)

# range 내에서 일정한 간격으로 num개 요소 할당하기
a6 = np.linspace(1, 10, 20) # [start, end] 구간을 num개로 일정하게 간극을 나눌 때, num의 default 값은 50
print(a6)
print("-"*20)

# 특정 값으로 numpy 배열 초기화
a7 = np.zeros((10, 10), np.int32)
print(a7)
print("-"*20)

a8 = np.ones(10, np.float64) # numpy.double과 완전히 같음
print(a8)
print("-"*20)

# 단위 행렬
a9 = np.eye(5, dtype=np.int32) # 세부 행렬 조정은 definition 참조
print(a9)
print("-"*20)

# numpy 배열 형변환
a10 = np.array(['1.5', '0.44', '3.14', '3.14599'])
print(a10)
print(a10.dtype) # 배열의 타입이 문자열인 경우, dtype은 f"U{숫자}" 형태가 됨. 이 때 U는 유니코드를 의미하고, {숫자}는 해당 배열 요소 중 가장 긴 요소의 길이
print(a10.astype(np.double))
print(a10.astype(np.double).dtype)
print("-"*20)

# 난수 배열 생성
a11 = np.random.rand(2,3)
print(a11)
print("-"*20)

a12 = np.random.rand(2, 3, 4) # 3차원 배열 [2][3][4]
print(a12)
print("-"*20)

a13 = np.random.randint(10, size=(3, 4)) # 0 ~ 9 사이의 정수
print(a13)
print("-"*20)


# 조건에 맞는 인덱스 배열을 각 차원별로 나누어 튜플 형태로 반환
indices = np.where(a13 >= 5)
print("5 이상인 값들의 인덱스:")
print(indices)
print("-"*20)
'''
(array([0, 0, 0, 0, 1], dtype=int64), array([0, 1, 2, 3, 2], dtype=int64))

# 좌측 배열은 조건을 만족하는 값들의 "행 인덱스" (첫 번째 차원)입니다.
# 우측 배열은 조건을 만족하는 값들의 "열 인덱스" (두 번째 차원)입니다.
# 이들을 한 쌍으로 읽으면 순서대로 조건에 부합한 값의 인덱스를 찾을 수 있습니다.
'''


# 조건에 따라 값 변환
print(np.where(a13 >= 5, '5이상', '5이하'))
print("-"*20)
'''
[['5이상' '5이하' '5이상' '5이하']
 ['5이상' '5이하' '5이하' '5이하']
 ['5이하' '5이상' '5이상' '5이하']]
'''


# 불리언 평가 값 기반 필터링
a14 = np.arange(10)
print(a14[a14 >= 5])
print("-"*20)


# 인덱스 기반 필터링
print(a14[np.where(a14 >= 2)])
print("-"*20)


# a14[np.array([2, 3, 4, 5, 6, 7, 8, 9])] 도 동일한 값을 출력한다.
print(a14[np.array([2, 3, 4, 5, 6, 7, 8, 9])])
print("-"*20)
