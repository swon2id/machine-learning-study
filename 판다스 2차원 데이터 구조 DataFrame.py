import pandas as pd

# DataFrame 객체의 개별 열과 행은 Series로 취급된다.
# 2차원 데이터 생성
data = {
    '이름': ['민지', '하니', '다니엘'],
    '수학': [95, 85, 75],
    '영어': [90, 88, 94]
}
df = pd.DataFrame(data)
print(df)
print("*" * 40)
'''
    이름  수학  영어
0   민지  95  90
1   하니  85  88
2  다니엘  75  94
'''


print(df.info())  # 데이터 타입 및 결측치 확인
print("*" * 40)
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   이름      3 non-null      object
 1   수학      3 non-null      int64 
 2   영어      3 non-null      int64 
dtypes: int64(2), object(1)
memory usage: 204.0+ bytes
None
'''

print("처음 n개 가져오기")
print(df.head(3))
print("="*80)

print("마지막 n개 가져오기")
print(df.tail(3))
print("="*80)

print("index 기반 가져오기")
print(df.iloc[0:2])
print("="*80)

print("무작위 n개 가져오기")
print(df.sample(1))
print("="*80)

print("행과 열 가져오기")
print(df.shape)
print("="*80)

print("컬럼 별 메타데이터(이름, 타입 등) 가져오기")
df.info()
print("="*80)

print(df.describe())  # 수치형 데이터 요약 통계 제공
print("*" * 80)
'''
         수학         영어
count   3.0   3.000000
mean   85.0  90.666667
std    10.0   3.055050
min    75.0  88.000000
25%    80.0  89.000000
50%    85.0  90.000000
75%    90.0  92.000000
max    95.0  94.000000
'''


print(df['수학'].describe())  # 특정 행 기준 요약 통계
print("*" * 40)
'''
count     3.0
mean     85.0
std      10.0
min      75.0
25%      80.0
50%      85.0
75%      90.0
max      95.0
Name: 수학, dtype: float64
'''


print(df.loc[0].describe())  # 특정 열 기준 요약 통계
print("*" * 40)
'''
count      3
unique     3
top       민지
freq       1
Name: 0, dtype: object
'''


print(df.describe(include='all'))
print("="*80)
'''
include='all':

이 옵션을 사용하면 모든 열에 대한 기술 통계 요약 결과가 포함됩니다.
숫자형 열의 통계 정보뿐만 아니라 문자열 열의 고유 값 수, 최빈값 등도 포함됩니다.

unique, top, freq는 문자 변수로만 계산하므로 숫자 변수에는 NaN이 출력됩니다.
반대로 숫자를 이용해 계산하는 요약 통계량은 숫자 변수에만 출력되고 문자 변수에는 NaN이 출력됩니다.

[출력]
         이름    수학         영어
count     3   3.0   3.000000
unique    3   NaN        NaN
top      민지   NaN        NaN
freq      1   NaN        NaN
mean    NaN  85.0  90.666667
std     NaN  10.0   3.055050
min     NaN  75.0  88.000000
25%     NaN  80.0  89.000000
50%     NaN  85.0  90.000000
75%     NaN  90.0  92.000000
max     NaN  95.0  94.000000
'''


# 일부 열만 출력
print(df[['이름', '수학']])
print("*" * 40)


# 특정 값 추출
print("특정 값 추출")
print(df['수학'])  # 수학 열 추출 (행 인덱스 포함)
print(df.loc[0])  # 0번 행 추출 (열 이름 포함)
print(df.loc[1, '영어'])  # 특정 단일 원자 값 추출
print("*" * 40)


# 정렬
print("정렬")
print(df.sort_values(by='수학'))  # 수학 점수 기준 ascending=True 정렬
print("*" * 40)
print(df.sort_values(['영어', '수학'])) # 영어가 우선순위가 더 높음
print("*" * 40)


# 새로운 값 추가
df['과학'] = [93, 89, 87]  # 새 열 추가
df.loc[3] = ['혜인', 92, 96, 94]  # 새 행 추가
print(df)
print("*" * 40)


# 기본 연산
print(df['수학'].sum())  # 합계
print(df['수학'].mean())  # 평균
print(df['수학'].max())  # 최대값
print(df['수학'].min())  # 최소값
print("*" * 40)


exam = pd.read_csv("exam.csv")
# query 메서드 사용하여 조건부 데이터 select
print("[ query 메서드 사용하여 조건부 데이터 select ]")
print("-" * 100, "1. nclass 열의 값이 1를 만족하는 데이터만 select", "-" * 100, sep='\n')
print(exam.query('nclass == 1'))

print("-" * 100, "2. math 열의 값이 50 초과인 데이터만 select", "-" * 100, sep='\n')
print(exam.query('math > 50'))

print("-" * 100, "3. 1반 이면서 수학 점수가 50점 이상인 데이터만 select", "-" * 100, sep='\n')
print(exam.query('nclass == 1 & math >= 50'))

print("-" * 100, "4. 1반이거나 2반이면서 모든 점수가 60점 이상인 데이터만 select", "-" * 100, sep='\n')
print(exam.query('(nclass == 1 | nclass == 2) & math >= 60 & english >= 60 & science >= 60'))

print("-" * 100, "5. query 이후 특정 column series 가져오기", "-" * 100, sep='\n')
print(exam.query('nclass == 1')['english'])

print()

# 특정 행과 열에 대해 빈도수 데이터 출력
print("[ 특정 행과 열에 대해 빈도수 데이터 출력 ]")
print("-" * 100, "1. 0번 행의 존재하는 모든 값 count", "-" * 100, sep='\n')
print(exam.loc[0].value_counts())

print("-" * 100, "2. math 열의 존재하는 모든 값 count with 오름차순 정렬", "-" * 100, sep='\n')
print(exam['math'].value_counts().sort_index())


# 열 제거
print(exam.drop(columns='math'))
print("*" * 40)

print(exam.drop(columns=['science', 'english']))
print("*" * 40)