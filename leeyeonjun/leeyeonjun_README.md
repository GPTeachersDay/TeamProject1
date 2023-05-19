# Notion
EDA 코드 보기(https://github.com/GPTeachersDay/TeamProject1/blob/main/leeyeonjun/EDA/eda_abalone.ipynb)
EDA 코드 보기(https://github.com/GPTeachersDay/TeamProject1/blob/main/leeyeonjun/EDA/eda_star.ipynb)
EDA 코드 보기(https://github.com/GPTeachersDay/TeamProject1/blob/main/leeyeonjun/EDA/eda_steel.ipynb)




<br><br><br><br>




# EDA
## EDA : 전복

#### features
Sex : 전복 성별(i=infant) / object  
Lenght : 전복 길이 / mm / float  
Diameter : 전복 둘레 / mm / float  
Height : 전복 길이 / mm / float  
Whole Weight : 전복 전체 무게 / grams / float  
Shucked Weight : 껍질을 제외한 무게 / grams / float  
Viscra Weight : 내장 무게 / grams / float  
Shell Weight : 껍질 무게 / grams / float  
Rings : 전복 나이 / int  


#### Domain
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


#### Workflow
1. 기본 통계량 확인
2. 결측치 확인
3. 상관 분석
4. 분포 확인
5. 카테고리별 분석
6. 피처 중요도 분석


#### Reference
- https://archive.ics.uci.edu/ml/datasets/abalone
- https://www.kaggle.com/code/anmolbajpai/abalone-age-prediction-regression
- https://rpubs.com/shihui17170153/abalone
- https://rstudio-pubs-static.s3.amazonaws.com/378381_1221e0d1034b4020a38a862a76890bb6.html




<br><br><br><br>




## EDA : 별

#### features
1. Mean of the integrated profile. (통합 프로파일의 평균.)
2. Standard deviation of the integrated profile. (통합 프로파일의 표준 편차.)
3. Excess kurtosis of the integrated profile. (통합 프로파일의 과도한 첨도.)
4. Skewness of the integrated profile. (통합 프로파일의 왜도.)
5. Mean of the DM-SNR curve. (DM-SNR 곡선의 평균.)
6. Standard deviation of the DM-SNR curve. (DM-SNR 곡선의 표준 편차.)
7. Excess kurtosis of the DM-SNR curve. (DM-SNR 곡선의 과도한 첨도.)
8. Skewness of the DM-SNR curve. (DM-SNR 곡선의 왜도.)
9. Class (타겟)


#### Domain
1. 펄서 스타란?:
펄서 스타는 매우 빠른 속도로 회전하는 중성자 별을 말합니다.
이들은 극도로 강한 자기장을 가지며, 이로 인해 빔 형태의 방출을 생성합니다.
이 방출이 지구를 향할 때마다 우리는 펄스 또는 신호를 감지할 수 있습니다.
이것이 펄서 스타라는 이름의 기원입니다.

2. 통합 프로파일 (Integrated Profile):
펄서의 펄스 프로파일은 펄서의 신호 강도를 시간에 따라 표시한 것입니다.
'통합 프로파일'은 이러한 펄스의 평균 프로파일을 말합니다.
평균, 표준 편차, 왜도, 첨도 등은 이러한 프로파일의 통계적 특성을 나타냅니다.

3. DM-SNR 곡선:
DM-SNR은 Dispersion Measure-Signal to Noise Ratio의 약자입니다.
Dispersion Measure는 펄서 신호가 우리에게 도달하는 데 걸리는 시간, 즉 신호의 지연을 측정합니다.
Signal to Noise Ratio는 신호의 강도와 배경 잡음의 강도 사이의 비율을 나타냅니다.
DM-SNR 곡선은 이 두 수치를 결합한 것입니다.

4. 분류 (Class):
이것은 우리가 예측하려는 목표 변수입니다.
여기서는 별이 펄서인지 아닌지를 나타냅니다.

#### Domain Advance
1. 통합 프로파일 (Integrated Profile):
통합 프로파일은 펄서 별에서 방출되는 전파 신호의 강도 분포를 시간에 따라 표현한 것입니다.
이는 펄서의 회전 주기를 거쳐서 관찰되는 신호 강도의 평균 패턴을 나타냅니다.
이 프로파일의 통계적 특성을 분석하면 펄서의 구조와 방출 메커니즘에 대한 중요한 통찰력을 얻을 수 있습니다.
그래서 평균, 표준편차, 왜도, 첨도 등의 통계적 특성이 피처로 사용됩니다.

2. DM-SNR 곡선:
DM-SNR은 Dispersion Measure (DM)와 Signal-to-Noise Ratio (SNR)를 결합한 것입니다.

- Dispersion Measure (DM)는 신호가 우리에게 도달하기까지 걸리는 시간, 즉 신호의 지연을 측정합니다.
이 지연은 신호가 펄서 별에서 지구까지 여행하면서 중간에 있는 자유 전자에 의해 발생합니다. 이 지연은 빛의 파장에 따라 다르며, 이를 이용하여 펄서 신호의 원본 스펙트럼을 재구성할 수 있습니다.

- Signal-to-Noise Ratio (SNR)는 신호의 강도와 배경 잡음의 강도 사이의 비율을 나타냅니다.
높은 SNR은 신호가 잡음보다 훨씬 강력하다는 것을 의미하며, 이는 신호 검출에 유리합니다.


#### Workflow
1. 기본 통계량 확인
2. 결측치 확인
3. 상관 분석
4. 분포 확인
5. 카테고리별 분석
6. 피처 중요도 분석


#### Reference
- https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate
- https://datauab.github.io/pulsar_stars/
- https://towardsdatascience.com/predicting-pulsar-stars-an-imbalanced-classification-task-comparing-bootstrap-resampling-to-smote-8cfbe037b807




<br><br><br><br>




## EDA : 강철판

#### features
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


#### target
1. Pastry : 페이스트리 결함, 이는 강판의 특정 부분이 너무 얇아져 결함이 발생한 경우를 의미할 수 있습니다.
2. Z_Scratch : Z 스크래치 결함, 강판에 'Z' 모양의 스크래치가 생긴 경우를 의미합니다.
3. K_Scatch : K 스크래치 결함, 강판에 'K' 모양의 스크래치가 생긴 경우를 의미합니다.
4. Stains : 얼룩 결함, 강판에 얼룩이나 불순물이 생긴 경우를 의미합니다.
5. Dirtiness : 더러움 결함, 강판이 더러워진 경우를 의미합니다.
6. Bumps : 돌기 결함, 강판에 돌기나 울퉁불퉁한 부분이 생긴 경우를 의미합니다.
7. Other_Faults : 기타 결함, 위의 카테고리에 속하지 않는 기타 다른 종류의 결함을 의미합니다.


#### Domain
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


#### Workflow
1. 기본 통계량 확인
2. 결측치 확인
3. 상관 분석
4. 분포 확인
5. 카테고리별 분석
6. 피처 중요도 분석


#### Reference
- https://www.kaggle.com/code/tahazafar/steel-plate-faults-data-analysis/notebook
- https://mindcompass.github.io/data_analysis/data_analysis(test2_1)/




<br><br><br><br>
