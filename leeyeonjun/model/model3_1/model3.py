import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from tensorflow import keras
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras import regularizers

class Model_3(keras.Model):
    def __init__(self
                , csv_path='mulit_classification_data.csv'
                , model_path='model3.h5'
                , TRAIN_RATIO=0.8):
        super().__init__()
        
        print(f'✅ 인스턴스 생성5')
        
        # 데이터셋 로드
        _, _, self.X_train, self.X_test, self.y_train, self.y_test, self.df, self.encoder = self.get_dataset(csv_path)
        
        # 모델생성
        self.model = self.get_model()
        self.model.load_weights(model_path)
    
    def get_dataset(self
                    , csv_path='mulit_classification_data.csv'
                    , TRAIN_RATIO=0.8):
        
        df = pd.read_csv(csv_path)
        df['Fault'] = df[['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']].idxmax(axis=1)
        
        # 라벨 인코딩
        encoder = LabelEncoder()
        df['Fault'] = encoder.fit_transform(df['Fault'])
        
        # 학습 데이터 분리
        X = df.drop(['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults', 'Fault'], axis=1)
        y = df['Fault']
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_RATIO, random_state = 83)
        
        return X, y, X_train, X_test, y_train, y_test, df, encoder
        
    
    def get_model(self):
        
        np.random.seed(83)
        tf.random.set_seed(83)
        
        units = [1024 for _ in range(8)]
        l2 = [0.001 + i * 0.0025 for i in range(8)]

        inputs = Input(shape=(len(self.X_train.keys()),))
        x = Dense(units=units[0], activation='relu', kernel_regularizer=regularizers.l2(l2[0]))(inputs)
        x = Dropout(0.2)(x)

        for i in range(1, 8):   
            dense = Dense(units=units[i], activation='relu', kernel_regularizer=regularizers.l2(l2[i]))
            y = dense(x)
            y = Dropout(0.2)(y)
            x = Add()([x, y])

        outputs = Dense(units=7, activation='softmax')(x) # 출력 유닛의 수를 클래스 수에 맞추고, softmax 활성화 함수를 사용

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
        
        # 입력된 데이터에 대하여 스케일링
        print(f'✅ input_list : {input_list}')
        input_scaled = self.scaler.transform([input_list])
        print(f'✅ input_scaled : {input_scaled}')

        # 타겟 예측
        y_RESULT = int(self.model.predict(input_scaled, verbose=0))
        # print(f'✅ y_RESULT : {y_RESULT}')
        
        return y_RESULT
        
