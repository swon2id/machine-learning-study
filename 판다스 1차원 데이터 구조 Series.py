"""
Pandas는 테이블 형태의 데이터를 쉽게 다룰 수 있도록 설계된 파이썬 데이터 분석 라이브러리입니다.
데이터 전처리, 탐색, 변환 및 시각화 등의 작업에 널리 사용됩니다.

numpy는 수치 연산에, pandas는 구조화된 데이터 처리와 탐색적 데이터 분석에 더 적합한데,
numpy와 비교하면 numpy는 단일 데이터 타입만 지원하고 배열 기반으로 연산 속도가 훨씬 빠르기 때문이다.
"""

import pandas as pd
import numpy as np


# Series는 pandas 1차원 데이터 구조로 각 데이터에 인덱스(라벨)가 붙은 형태이다.
s1 = pd.Series([10, 20, 30, 40, 50])
print(s1)
print(s1.index)
print(s1.values)
print("*" * 40)
"""
0    10
1    20
2    30
3    40
4    50
dtype: int64
RangeIndex(start=0, stop=5, step=1)
[10 20 30 40 50] => numpy 배열 출력 형식과 동일
"""


# Series에 index 값으로 시퀀스 자료형을 전달하여 사용자 정의 인덱스를 넣을 수 있다. 이 때 길이는 서로 같아야 한다.
index_date = ('2023-09-15', '2023-09-16', '2023-09-17', '2023-09-18')
s4 = pd.Series([200, 195, np.nan, 205], index=index_date)
print(s4)
print("*" * 40)


# 또는 애초에 Series에 dict 형태로 데이터를 넣어 label, value를 넣을 수도 있다.
s5 = pd.Series({'국어': 100, '영어': 95, '수학': 90})
print(s5)
print("*" * 40)


# 인덱스에 사용할 시퀀스 데이터로 DatetimeIndex를 사전에 생성할 수 있다.
datetime_index1 = pd.date_range(start='2023-09-16', end='2023-09-18')
print(datetime_index1)
print(pd.Series([1, 2, 3], index=datetime_index1))
print("*" * 40)


# end 속성 대신에 값의 갯수로 periods와 단위로 freq를 사용할 수 있다. freq는 'D' 'W' 'M' 등이 있다.
datetime_index2 = pd.date_range(start='2023-09-16', periods=4, freq='W') # 주 단위로 얻기
print(datetime_index2)
print("*" * 40)


# 여러 타입의 데이터가 들어가면 dtype은 object가 된다.
print(pd.Series([10.0, "20", 30, 40, 50]))
print("*" * 40)
"""
0    10.0
1      20
2      30
3      40
4      50
dtype: object
"""


# series가 모두 숫자인 경우 np.nan으로 not a number 값을 넣을 수 있다.
# 단점은 np.nan가 float로 구현되어 dtype을 object로 설정하지 않으면 float64로 데이터 타입이 고정된다는 점이다.
print(pd.Series([np.nan, 10, 2], dtype='object'))
print("*" * 40)
"""
0     NaN
1    10.0
2     2.0
dtype: float64
"""


# None은 데이터가 모두 숫자 형태이면, NaN으로 처리되지만, 문자가 섞이면 None으로 표시된다.
print(pd.Series([None, 1, 2, 3, None]))
print("*" * 40)
"""
0    NaN
1    1.0
2    2.0
3    3.0
4    NaN
dtype: float64
"""


print(pd.Series([None, '1', 1, 2, 3, None]))
print("*" * 40)
"""
0    None
1       1
2       1
3       2
4       3
5    None
dtype: object
"""

# 결측 값 표시를 위해 pd.NA를 사용할 수도 있다.
print(pd.Series([pd.NA, '1', 1, 2, 3, pd.NA]))
print("*" * 40)
'''
0    <NA>
1       1
2       1
3       2
4       3
5    <NA>
dtype: object
'''

# 정렬
print("정렬")
print(pd.Series([3,1,2,3,6,5,8,4]).sort_values(ascending=False)) # ascending 기본 값은 True


# 필터링 할 때
# Series는 numpy 배열처럼 취급할 수 있기 때문에 numpy.where 메서드를 사용할 수 있고
# Series.isin 메서드도 사용할 수 있다.
s = pd.Series([1, 2, 3, 4, 5, 6, 7])

print("[numpy.where로 필터링]")
print(np.where(s >= 3)) # numpy.ndarray로 구성된 튜플 반환

print("\n[Series.isin으로 필터링]")
values_to_check = [2, 4, 6]
print(s[s.isin(values_to_check)]) # 필터링된 Series 반환
print("*" * 40)


# 빈도수 출력
print(pd.Series([pd.NA, '1', 1, 1, 2, 3, pd.NA]).value_counts())
print("*" * 40)


# 데이터 삭제
print(pd.Series(np.arange(10)).drop([0, 1, 2]))