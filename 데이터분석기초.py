# 데이터분석: 데이터에서 유의미한 정보를 추출하고, 이를 통해 인사이트를 도출하는 과정
# 데이터 분석은 다양한 통계적 기법과 머신러닝 알고리즘을 사용하여 데이터에서 패턴을 찾고,
# 예측하거나 의사결정을 위한 정보를 제공합니다. 분석 결과는 시각화하여 더 쉽게 이해할 수 있도록 합니다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


mpg = pd.read_csv('mpg.csv')
print(mpg)
print("="*80)


# 총점 열 추가 후 히스토그램 출력 (numeric 데이터만 가능)
mpg['total'] = (mpg['cty'] + mpg['hwy']) / 2
mpg['total'].plot.hist()
plt.show()


# 통과 열 추가 후 통과 빈도 Series 막대 그래프 출력
mpg['pass'] = np.where(mpg['total'] >= 20, 'y', 'n')
count_pass = mpg['pass'].value_counts() # 연비 합격 막대 그래프 만들기
count_pass.plot.bar(rot = 0) # x축 value 회전 값 0도로 설정 (기본 -90)
plt.show()

print(mpg['category'].isin(['compact', 'subcompact', '2seater']))
print(np.where(mpg['category'].isin(['compact', 'subcompact', '2seater'])))