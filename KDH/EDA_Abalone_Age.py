# Regression 모델
""" features
Sex : 전복 성별 (object)
Lenght : 전복 길이 (float)
Diameter : 전복 둘레 (float)
Height : 전복 키 (float)
Whole : Weight : 전복 전체 무게 (float)
Shucked Weight : 껍질을 제외한 무게 (float)
Viscra Weight : 내장 무게 (float)
Shell Weight : 껍질 무게 (float)
Rings(Target) : 전복 나이 (int)
"""

""" Domain 지식

1. 전복의 나이 판단:
전복의 나이는 보통 그것의 껍질에 있는 고리의 수를 세어 판단합니다.
이것이 'Rings' 피처가 가리키는 것입니다.
따라서 'Rings' 피처는 전복의 나이를 직접적으로 예측하는 데 중요한 변수가 될 수 있습니다.

2. 전복의 성장:
전복은 자신의 생명 동안 계속 성장합니다.
따라서 길이, 둘레, 높이, 그리고 무게와 같은 피처들은 전복의 나이와 밀접한 관련이 있을 수 있습니다.
더 크고 무거운 전복일수록 일반적으로 더 오래된 전복일 것입니다.

3. 전복의 성별:
전복은 성별에 따라 다른 성장 패턴과 생리적 특성을 가질 수 있습니다.
예를 들어, 일부 연구에서는 암컷 전복이 수컷 전복보다 더 빨리 성장하고 더 큰 크기에 도달할 수 있다는 것을 발견했습니다.
따라서 'Sex' 피처는 전복의 나이를 예측하는 데 중요한 역할을 할 수 있습니다.

4. 전복의 무게:
전복의 껍질을 제외한 무게 ('Shucked Weight'), 내장 무게 ('Viscera Weight'),
그리고 껍질 무게 ('Shell Weight')는 전복의 전체 무게 ('Whole Weight')를 구성합니다.
이러한 피처들은 전복의 성장과 밀접한 관련이 있을 수 있으며, 따라서 나이 예측에 중요할 수 있습니다.
"""

"""EDA 워크플로우
1. 기본 통계량 확인
2. 결측치 확인
3. 상관 분석
4. 분포 확인
5. 카테고리별 분석
6. 피처 중요도 분석
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Dataset/Regression_data.csv")
df.info()

le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])

# 상관계수
correlation_matrix = df.corr()
print(round(correlation_matrix, 2))

corr_data = round(correlation_matrix, 2)
corr_data.to_excel('Abalone_corr.xlsx', index=True)