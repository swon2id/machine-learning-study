import numpy as np

arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])

# 요소별 덧셈
result = arr1 + arr2
print(result)

# 요소별 곱셈
result = arr1 * arr2
print(result)

# 요소별 거듭제곱
result = arr1 ** 2
print(result)

# 요소별 비교 연산 : boolean 반환
arr3 = np.array([10, 20, 30, 40])
print(arr3 > 20)

arr4 = np.arange(5)
print(f"합계 : {arr4.sum()}, 평균 : {arr4.mean()}")
print(f"표준편차 : {arr4.std()}, 분산 : {arr4.var()}")
print(f"최솟값 : {arr4.min()}, 최댓값 : {arr4.max()}")

# 행렬 연산
matrix1 = np.array([[1,2], [3,4]])
matrix2 = np.array([[5,6], [7,8]])
print(np.add(matrix1, matrix2))  # 행렬 덧셈
print(np.dot(matrix1, matrix2))  # 행렬 곱셈
print(np.linalg.inv(matrix1))  # 역행렬

# 정렬 및 탐색
x = np.array([9,8,7,2,3,4,6])
print(np.sort(x))  # 오름차순 정렬
print(np.argsort(x))  # 정렬된 인덱스 반환

# 인덱싱 / 슬라이싱
arr = np.array([1,2,3,4,5])
print(np.power(arr, 2))
print(arr ** 2)

#인덱싱
print(arr[0])
print(arr[2])

# 슬라이싱
print(arr[1:4]) #[2,3,4]

"""
Universal 함수: 각 원소에 대해 동일한 연산을 적용

[산술 연산]
np.add(numpy_array, 5) == numpy_array + 5
np.subtract(numpy_array, 5) == numpy_array + 5
np.multiply(numpy_array, 5) == numpy_array * 5
np.divide(numpy_array, 5) == numpy_array / 5
np.power(numpy_array, 2) == numpy_array ** 2

[삼각 함수]
np.sin()
np.cos()
np.tan()

[지수와 로그 함수]
np.exp()
np.log()
np.log10()

[집계 함수]
np.sum()
np.mean()
np.max()
np.min()

[논리 함수]
np.logical_and()
np.logical_or()
np.logical_not()
"""

"""
[브로드 캐스팅]
배열의 크기가 서로 다른 경우의 연산 수행 시 
a는 1차원 배열이고, b는 2차원 배열이므로 브로드캐스팅에 의해 두 배열은 동일한 차원 크기로 확장됩니다.
a의 형태는 (3,), b의 형태는 (3, 1)입니다. 따라서 a가 열 방향으로 확장되어 (3, 3) 형태로 변환됩니다. b는 각 열에 대해 값이 반복되어 (3, 3) 형태로 확장됩니다.
a + b의 결과:

브로드캐스팅을 사용한 연산 결과는 a + b가 배열의 각 항목을 더하는 형태로 작동합니다. a가 (3,)에서 (3, 3)으로 확장되고, b가 (3, 1)에서 (3, 3)으로 확장되어 각 요소별로 덧셈을 수행합니다.
"""
a = np.array([1, 2, 3])  # 1차원 배열
b = np.array([[4], [5], [6]])  # 2차원 배열 (3 X 1)

c = a + b
print(c)  # 브로드캐스팅 결과

a = np.array(([1, 2, 3], [1, 2, 3], [1, 2, 3]))  # 2차원 배열 (3 X 3)
b = np.array(([4, 4, 4], [5, 5, 5], [6, 6, 6]))  # 2차원 배열 (3 X 3)
c = a + b
print(c)  # 직접 더한 결과

