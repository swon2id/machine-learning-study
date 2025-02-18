from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

plt.plot(tuple(range(10, 50, 10))) # y축, x축은 지정되지 않아 0에서 시작하는 값으로 자동 설정
plt.show()

# 동시에 여러 데이터 plot
plt.title('color') # 제목 설정
plt.plot([10, 20, 30, 40], linestyle='--', color='skyblue', label='skyblue')
plt.plot([40, 30, 20, 10], linestyle=':', color='pink', label='pink')
plt.legend() # 범례 표시
plt.show()

# 0차원 점 표시
plt.plot([10, 20, 30, 40], 'r.', label='Circle Marker')
plt.plot([40, 30, 20, 10], 'g^', label='Triangle Marker')
plt.legend()
plt.show()

import csv
f = open('seoul.csv', encoding='cp949') # cp949 형식은 그냥 제공하는 데이터와 호환성 때문에 쓸뿐
data = csv.reader(f)
header = next(data)
max_temp = -999 # 최고 기온값을 지정할 변수
max_date = '' # 최고 기온이 가장 높았던 날짜를 지정할 변수

for row in data :
    if row[-1] == '':
        row[-1] = -999 # -999를 넣어 빈 문자열이 있던 자리라고 표시
    row[-1] = float(row[-1])
    #print(row)
    if max_temp < row[-1]:
        max_date = row[0]
        max_temp = row[-1]
f.close()
# print(max_date, max_temp)
print(f"기상 관측이래 서울의 최고 기온이 가장 높은던 날은 {max_date} 이며, {max_temp}도 였습니다.")

with open("gender.csv", encoding='cp949') as file:
    data = csv.reader(file)
    m = None
    f = None

    for row in data:
        if '신도림' in row[0]:
            m = list(map(int, row[3:104]))
            f = list(map(int, row[106:]))
            print(m)
            print(f)

    file.close()

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.title(f'신도림 지역의 남녀 성별 인구 분포')
plt.barh(range(101), m, label='남성')
plt.barh(range(101), f, label='여성')
plt.legend()
plt.show()