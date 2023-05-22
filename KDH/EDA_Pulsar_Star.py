# Binary Classification 모델
""" features
1. Mean of the integrated profile. (통합 프로파일의 평균.)
2. Standard deviation of the integrated profile. (통합 프로파일의 표준 편차.)
3. Excess kurtosis of the integrated profile. (통합 프로파일의 과도한 첨도.)
4. Skewness of the integrated profile. (통합 프로파일의 왜도.)
5. Mean of the DM-SNR curve. (DM-SNR 곡선의 평균.)
6. Standard deviation of the DM-SNR curve. (DM-SNR 곡선의 표준 편차.)
7. Excess kurtosis of the DM-SNR curve. (DM-SNR 곡선의 과도한 첨도.)
8. Skewness of the DM-SNR curve. (DM-SNR 곡선의 왜도.)
9. Class (타겟)
"""

""" Domain 지식

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
"""

""" Domain 지식 심화
1. 통합 프로파일 (Integrated Profile):
통합 프로파일은 펄서 별에서 방출되는 전파 신호의 강도 분포를 시간에 따라 표현한 것입니다.
이는 펄서의 회전 주기를 거쳐서 관찰되는 신호 강도의 평균 패턴을 나타냅니다.
이 프로파일의 통계적 특성을 분석하면 펄서의 구조와 방출 메커니즘에 대한 중요한 통찰력을 얻을 수 있습니다.
그래서 평균, 표준편차, 왜도, 첨도 등의 통계적 특성이 피처로 사용됩니다.

2. M-SNR 곡선:
DM-SNR은 Dispersion Measure (DM)와 Signal-to-Noise Ratio (SNR)를 결합한 것입니다.

- Dispersion Measure (DM)는 신호가 우리에게 도달하기까지 걸리는 시간, 즉 신호의 지연을 측정합니다.
이 지연은 신호가 펄서 별에서 지구까지 여행하면서 중간에 있는 자유 전자에 의해 발생합니다. 이 지연은 빛의 파장에 따라 다르며, 이를 이용하여 펄서 신호의 원본 스펙트럼을 재구성할 수 있습니다.

- Signal-to-Noise Ratio (SNR)는 신호의 강도와 배경 잡음의 강도 사이의 비율을 나타냅니다.
높은 SNR은 신호가 잡음보다 훨씬 강력하다는 것을 의미하며, 이는 신호 검출에 유리합니다.
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

df = pd.read_csv("Dataset/binary_classification_data.csv")
df.info()

# 상관계수
correlation_matrix = df.corr()
print(round(correlation_matrix, 2))

corr_data = round(correlation_matrix, 2)
corr_data.to_excel('Pulsar_corr.xlsx', index=True)

import tensorflow as tf
tf.config.list_physical_devices('GPU')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# TensorFlow에서 사용 가능한 GPU 장치 목록을 출력합니다.
gpu_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", len(gpu_devices))

# GPU가 사용 가능한 경우, GPU 장치를 설정합니다.
if len(gpu_devices) > 0:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    tf.config.set_visible_devices(gpu_devices[0], 'GPU')