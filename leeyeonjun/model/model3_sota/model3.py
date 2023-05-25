import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras import regularizers
from keras.layers import Input, Dense, Dropout, Add
from keras.models import Model

from imblearn.over_sampling import SMOTE

from tensorflow import keras
import tensorflow as tf

class Model_3(keras.Model):
    def __init__(self
                , csv_path='mulit_classification_data.csv'
                , model_path='model3.h5'
                , TRAIN_RATIO=0.8):
        super().__init__()
        
        # print(f'✅ 인스턴스 생성1')
        
        # 데이터셋 로드
        self.df = pd.read_csv(csv_path)
        
        # 인코딩 방식 변경
        # idxmax 함수는 각 행의 최대값을 가진 열의 인덱스를 반환한다. 따라서 원핫인코딩된 피쳐를 하나의 카테고리 변수로 복원할 수 있음
        self.df['Fault'] = self.df[['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']].idxmax(axis=1)
        
        # 라벨 인코딩
        self.encoder = LabelEncoder()
        self.df['Fault'] = self.encoder.fit_transform(self.df['Fault'])
        
        # 이상치 제거
        self.df = self.df[~((self.df['Pixels_Areas'] > 35000) |
                (self.df['X_Perimeter'] > 2000) |
                (self.df['Y_Perimeter'] > 2500) |
                (self.df['Sum_of_Luminosity'] > 0.5e7))]

        # 'TypeOfSteel_A300'과 'TypeOfSteel_400'로 나누어진 특성을 하나로 통합
        self.df['TypeOfSteel'] = self.df['TypeOfSteel_A300']
        self.df.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400'], axis=1, inplace=True)

        # 다중공선성 문제 해결을 위해 상관관계가 높은 feature 확인 후 제거 (상관계수 절대값 0.9이상)
        # 'Pixels_Areas'는 특성중요도가 높으므로 활용하고 상관관계가 높은 나머지 특성들 제거
        self.df.drop(['Sum_of_Luminosity', 'X_Perimeter', 'Y_Perimeter'], axis=1, inplace=True)

        # 로그 스케일링(관측치 측정 범위가 넓음)
        self.df['Log_Pixels_Areas'] = np.log(self.df['Pixels_Areas'])
        self.df.drop(['Pixels_Areas'], axis=1, inplace=True)

        # 중요도 낮은 특성 제거(머신러닝 분류모델에서 사용하는 기법인데 딥러닝에도 적용이 가능한가? -> 우선 모델 성능 개선은 되었음, 더 알아보면 좋을 것 같음)
        self.df.drop(['Outside_Global_Index', 'SigmoidOfAreas', 'Log_Y_Index'], axis=1, inplace=True)

        # 학습 데이터 분리
        self.X = self.df.drop(['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults', 'Fault'], axis=1)
        self.y = self.df['Fault']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=TRAIN_RATIO, random_state = 83)
        
        # 오버샘플링(학습데이터만)
        # SMOTE 적용
        self.sm = SMOTE(random_state=2)
        self.X_train_res, self.y_train_res = self.sm.fit_resample(self.X_train, self.y_train.ravel())
        
        # 표준화(MinMaxScaling)
        # 스케일링할 피처 선택
        self.scaling_features = self.X_train.columns

        # 스케일링
        self.scaler = MinMaxScaler()
        self.X_train_scaled = self.X_train.copy()  # 원본 데이터 복사
        self.X_test_scaled = self.X_test.copy()    # 원본 데이터 복사
        self.X_train_scaled[self.scaling_features] = self.scaler.fit_transform(self.X_train[self.scaling_features])
        self.X_test_scaled[self.scaling_features] = self.scaler.transform(self.X_test[self.scaling_features])
        
        # 모델 가중치 로드
        self.model = self.get_model()
        self.model.load_weights(model_path)
    
    def get_model(self, dout=0.1, output_bias=None):
        
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)
        
        # 스킵 연결 모델(Skip Connection)
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # 노드 개수, L2 설정
        units = [256 for _ in range(8)]
        l2 = [0.001 + i * 0.0025 for i in range(8)]
        
        # 입력층을 정의하며, 입력의 형태는 훈련 데이터의 특성 수에 따라 결정
        inputs = Input(shape=(len(self.X_train.keys()),))
        
        # 첫 번째 Dense 층을 만들고 ReLU 활성화 함수를 사용, L2 정규화 적용
        x = Dense(units=units[0], activation='relu', kernel_regularizer=regularizers.l2(l2[0]))(inputs)
        x = Dropout(0.2)(x)

        # 이후 7개의 Dense 층을 스킵 연결을 가지도록 생성
        for i in range(1, 8):   
            dense = Dense(units=units[i], activation='relu', kernel_regularizer=regularizers.l2(l2[i]))
            y = dense(x)
            y = Dropout(0.2)(y)
            
            # 현재 층의 출력(y)과 이전 층의 출력(x)을 더하여 스킵 연결 구현
            x = Add()([x, y])

        # 출력층을 정의, 유닛의 수는 클래스의 수와 동일하며, softmax 활성화 함수를 사용
        outputs = Dense(units=7, activation='softmax')(x)

        # 모델을 생성, 입력과 출력을 지정
        model = Model(inputs=inputs, outputs=outputs)
        
        # 옵티마이저와 손실 함수 설정
        optimizer = tf.keras.optimizers.Adam(
                                            learning_rate=0.001,
                                            beta_1=0.9,
                                            beta_2=0.999,
                                            epsilon=1e-08
                                            )

        model.compile(loss='sparse_categorical_crossentropy',  # 손실함수를 다중 클래스 분류에 적합한 형태로 변경
                    optimizer=optimizer,
                    metrics=['accuracy'])
        
        return model
    
    def input_predict(self, input_list):

        # 타겟 예측
        result_list = self.model.predict([input_list], verbose=0)
        # print(f"✅result_list : {result_list}")
        
        result_tup = [(i,v) for i,v in enumerate(result_list[0])]
        result_tup.sort(key=lambda x:-x[1])
        
        y_label_1 = result_tup[0][0]
        y_label_2 = result_tup[1][0]
        y_RESULT_1 = self.encoder.inverse_transform([y_label_1])[0]
        y_RESULT_2 = self.encoder.inverse_transform([y_label_2])[0]
        # print(f"y_RESULT : {y_RESULT_1}")
        # print(f"y_RESULT : {y_RESULT_2}")
        
        y_rate_1 = round(result_list[0][y_label_1]*100, 2)
        y_rate_2 = round(result_list[0][y_label_2]*100, 2)
        
        template = f"""
        <div class="container text-center border border-3 rounded">
            <h5 class="p-2 mb-1">
                🔎 강판에는<br>
                {y_rate_1}% 확률로 '<b>{y_RESULT_1}</b>' 결합<br>
                {y_rate_2}% 확률로 '<b>{y_RESULT_2}</b>' 결합<br>
                이 있습니다. 😃
            </h5>
        </div>
        """

        return template
        
