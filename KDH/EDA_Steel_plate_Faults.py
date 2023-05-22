# Multiple Classification 모델
""" features
1. X_Minimum : X의 최소값
2. X_Maximum : X의 최대값
3. Y_Minimum : Y의 최소값
4. Y_Maximum : Y의 최대값
5. Pixels_Areas : 픽셀 영역 (오류가 발생한 강판의 영역을 픽셀로 표시한 크기)
6. X_Perimeter : X 주변의 길이 (오류가 발생한 강판의 X 방향 주변 길이)
7. Y_Perimeter : Y 주변의 길이 (오류가 발생한 강판의 Y 방향 주변 길이)
8. Sum_of_Luminosity : 휘도의 합 (오류가 발생한 강판 영역의 전체 휘도)
9. Minimum_of_Luminosity : 최소 휘도 (오류가 발생한 강판 영역의 최소 휘도)
10. Maximum_of_Luminosity : 최대 휘도 (오류가 발생한 강판 영역의 최대 휘도)
11. Length_of_Conveyer : 컨베이어의 길이
12. TypeOfSteel_A300 : A300 형 강판 여부
13. TypeOfSteel_A400 : A400 형 강판 여부
14. Steel_Plate_Thickness : 강판의 두께
15. Edges_Index : 가장자리 지수 (오류가 발생한 강판의 가장자리 형태를 나타내는 지표)
16. Empty_Index : 빈 공간 지수 (오류가 발생한 강판의 빈 공간을 나타내는 지표)
17. Square_Index : 정사각형 지수 (오류가 발생한 강판의 정사각형 형태를 나타내는 지표)
18. Outside_X_Index : 외부 X 지수 (오류가 발생한 강판의 X 방향 외부 형태를 나타내는 지표)
19. Edges_X_Index : 가장자리 X 지수 (오류가 발생한 강판의 X 방향 가장자리를 나타내는 지표)
20. Edges_Y_Index : 가장자리 Y 지수 (오류가 발생한 강판의 Y 방향 가장자리를 나타내는 지표)
21. Outside_Global_Index : 전체 외부 지수 (오류가 발생한 강판의 전체 외부 형태를 나타내는 지표)
22. LogOfAreas : 영역의 로그값
23. Log_X_Index : X 지수의 로그값
24. Log_Y_Index : Y 지수의 로그값
25. Orientation_Index : 방향 지수 (오류가 발생한 강판의 방향을 나타내는 지표)
26. Luminosity_Index : 휘도 지수 (오류가 발생한 강판의 휘도를 나타내는 지표)
27. SigmoidOfAreas : 영역의 시그모이드 함수 값 (이 함수는 보통 0과 1 사이의 결과를 내며, 여기서는 오류 영역의 특성을 나타낼 수 있습니다)
"""

"""Onehotcoded features
1. Pastry : 페이스트리 결함, 이는 강판의 특정 부분이 너무 얇아져 결함이 발생한 경우를 의미할 수 있습니다.
2. Z_Scratch : Z 스크래치 결함, 강판에 'Z' 모양의 스크래치가 생긴 경우를 의미합니다.
3. K_Scatch : K 스크래치 결함, 강판에 'K' 모양의 스크래치가 생긴 경우를 의미합니다.
4. Stains : 얼룩 결함, 강판에 얼룩이나 불순물이 생긴 경우를 의미합니다.
5. Dirtiness : 더러움 결함, 강판이 더러워진 경우를 의미합니다.
6. Bumps : 돌기 결함, 강판에 돌기나 울퉁불퉁한 부분이 생긴 경우를 의미합니다.
7. Other_Faults : 기타 결함, 위의 카테고리에 속하지 않는 기타 다른 종류의 결함을 의미합니다.

"""

""" Domain 지식
강판 결함 분석은 제조 공정에서 중요한 부분입니다.
강판은 건축, 자동차, 선박, 가전제품 등 다양한 산업에서 사용되는 주요 소재이기 때문에, 그 품질은 최종 제품의 품질에 직접적인 영향을 미칩니다.
따라서 강판의 결함을 정확하게 감지하고 분류하는 것은 제조 효율성, 안전성, 그리고 경제성을 높이는 데 매우 중요합니다.

1. 강판의 특성:
강판의 결함은 종류와 크기에 따라 다양하게 나타날 수 있습니다.
결함의 유형에는 표면상의 불규칙한 부분, 두께의 불규칙, 재질 결함, 구조적 결함 등이 있을 수 있습니다.

2. 제조 공정:
강판 제조 공정에서의 다양한 단계에서 결함이 발생할 수 있습니다.
예를 들어, 제련, 주조, 압연, 열처리, 표면 마무리 등의 단계에서 결함이 발생할 수 있으며, 이는 최종 제품의 품질에 영향을 미칩니다.

3. 감지 기술:
강판 결함 감지는 시각 검사, 자동 결함 감지 시스템, 비파괴 검사 등 다양한 방법으로 수행될 수 있습니다.
머신 러닝 및 AI는 이러한 결함 감지 작업을 자동화하고 정확성을 높이는 데 도움이 될 수 있습니다.

4. 분석 및 품질 관리:
결함 분석은 결함의 원인을 찾고, 제조 공정을 개선하여 향후 결함을 방지하는 데 사용됩니다.
또한, 강판의 품질은 ISO 등의 국제 표준에 따라 평가되고 관리됩니다.
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

df = pd.read_csv("Dataset/mulit_classification_data.csv")
df.info()

# 상관계수
correlation_matrix = df.corr()
print(round(correlation_matrix, 2))

corr_data = round(correlation_matrix, 2)
corr_data.to_excel('Steel_corr.xlsx', index=True)