import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
import tensorflow as tf

class Model_2():
    def __init__(self
                , csv_path='Regression_data.csv'
                , model_path='model2.h5'
                , TRAIN_RATIO=0.8):
        
        # print(f"✅ 현재 위치 : {os.getcwd()}")
        # print(f'✅ 인스턴스 생성2')
        
        # 데이터셋 로드
        self.df = pd.read_csv(csv_path)
        
        # 학습 데이터 분리
        self.X = self.df.drop('target_class', axis=1)
        self.y = self.df['target_class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=TRAIN_RATIO, random_state = 83)
        
        # 표준화
        self.scaler     = StandardScaler() 
        self.X_train    = self.scaler.fit_transform(self.X_train)
        self.X_test     = self.scaler.transform(self.X_test)
        self.X_train    = pd.DataFrame(self.X_train, columns=self.X.columns)
        self.X_test     = pd.DataFrame(self.X_test, columns=self.X.columns)
        
        # 모델생성
        self.model = self.get_model()
        self.model.load_weights(model_path)
    
    def get_model(self, dout=0.1, output_bias=None):
        
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)
        
        
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)
        
        regularizer=keras.regularizers.L1L2(l1=0.001, l2=0.001)
        
        model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(12, activation='relu', input_shape=(self.X_train.shape[1],)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(dout),
                tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=regularizer),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(dout),
                tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=regularizer),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(dout),
                tf.keras.layers.Dense(6, activation='relu', kernel_regularizer=regularizer),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        
        METRICS = [
                keras.metrics.TruePositives(name='tp'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.TrueNegatives(name='tn'),
                keras.metrics.FalseNegatives(name='fn'), 
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
                ]
        
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=METRICS
            )
        return model
    
    def input_predict(self, request):
        
        IP_Mean = float(request.POST['IP_Mean'])
        IP_SD = float(request.POST['IP_SD'])
        IP_EK = float(request.POST['IP_EK'])
        IP_SK = float(request.POST['IP_SK'])
        Curve_Mean = float(request.POST['Curve_Mean'])
        Curve_SD = float(request.POST['Curve_SD'])
        Curve_EK = float(request.POST['Curve_EK'])
        Curve_SK = float(request.POST['Curve_SK'])
        
        input_list = [IP_Mean, IP_SD, IP_EK, IP_SK, Curve_Mean, Curve_SD, Curve_EK, Curve_SK]
        
        # 입력된 데이터에 대하여 스케일링
        input_scaled = self.scaler.transform([input_list])

        # 타겟 예측
        y_RESULT = int(self.model.predict(input_scaled, verbose=0))
        
        if y_RESULT:
            template = f"""
            <div class="container text-center border border-3 rounded">
                <h5 class="p-2 mb-1">
                    😃 맥동성이 맞습니다 🎉
                </h5>
            </div>
            """
        else:
            template = f"""
            <div class="container text-center border border-3 rounded">
                <h5 class="p-2 mb-1">
                    😥 맥동성이 아니에요 🔍
                </h5>
            </div>
            """
        # print(f'✅ y_RESULT : {y_RESULT}')
        return y_RESULT, template
        
