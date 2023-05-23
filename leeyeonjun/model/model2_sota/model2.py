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
        
        print(f'✅ 인스턴스 생성2')
        
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
        self.model = self.get_model(len(self.X_train.columns))
        self.model.load_weights(model_path)
    
    def get_model(self, shape, dout=0.001, output_bias=None):
        
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)
        
        regularizer=keras.regularizers.L1L2(l1=0.001, l2=0.001)
    
        inputs  = keras.Input(shape=shape)

        dense1  = keras.layers.Dense(32, kernel_regularizer=regularizer)(inputs)
        norm1   = keras.layers.BatchNormalization()(dense1)
        relu1   = keras.layers.Activation('relu')(norm1)
        dout1   = keras.layers.Dropout(dout)(relu1)
        
        dense2  = keras.layers.Dense(16, kernel_regularizer=regularizer)(dout1)
        norm2   = keras.layers.BatchNormalization()(dense2)
        relu2   = keras.layers.Activation('relu')(norm2)
        dout2   = keras.layers.Dropout(dout)(relu2)
        
        dense3  = keras.layers.Dense(8, kernel_regularizer=regularizer)(dout2)
        norm3   = keras.layers.BatchNormalization()(dense3)
        relu3   = keras.layers.Activation('relu')(norm3)
        dout3   = keras.layers.Dropout(dout)(relu3)
        
        dense4  = keras.layers.Dense(6, kernel_regularizer=regularizer)(dout3)
        norm4   = keras.layers.BatchNormalization()(dense4)
        relu4   = keras.layers.Activation('relu')(norm4)
        dout4   = keras.layers.Dropout(dout)(relu4)
        
        dense5  = keras.layers.Dense(4, kernel_regularizer=regularizer)(dout4)
        norm5   = keras.layers.BatchNormalization()(dense5)
        relu5   = keras.layers.Activation('relu')(norm5)
        dout5   = keras.layers.Dropout(dout)(relu5)
        
        concat1  = keras.layers.Concatenate(axis=1)([dout5, dout4])
        dense6  = keras.layers.Dense(6, kernel_regularizer=regularizer)(concat1)
        norm6   = keras.layers.BatchNormalization()(dense6)
        relu6   = keras.layers.Activation('relu')(norm6)
        dout6   = keras.layers.Dropout(dout)(relu6)
        
        concat2  = keras.layers.Concatenate(axis=1)([dout6, dout3])
        dense7  = keras.layers.Dense(8, kernel_regularizer=regularizer)(concat2)
        norm7   = keras.layers.BatchNormalization()(dense7)
        relu7   = keras.layers.Activation('relu')(norm7)
        dout7   = keras.layers.Dropout(dout)(relu7)
        
        concat3  = keras.layers.Concatenate(axis=1)([dout7, dout2])
        dense8  = keras.layers.Dense(16, kernel_regularizer=regularizer)(concat3)
        norm8   = keras.layers.BatchNormalization()(dense8)
        relu8   = keras.layers.Activation('relu')(norm8)
        dout8   = keras.layers.Dropout(dout)(relu8)
        
        concat4  = keras.layers.Concatenate(axis=1)([dout8, dout1])
        dense9  = keras.layers.Dense(32, kernel_regularizer=regularizer)(concat4)
        norm9   = keras.layers.BatchNormalization()(dense9)
        relu9   = keras.layers.Activation('relu')(norm9)
        dout9   = keras.layers.Dropout(dout)(relu9)
        
        outputs = keras.layers.Dense(1, activation='sigmoid'
                                 , bias_initializer=output_bias)(dout9)
        model   = keras.Model(inputs, outputs, name='Star_Model')
        
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
    
    def input_predict(self, input_list):
        
        # 입력된 데이터에 대하여 스케일링
        input_scaled = self.scaler.transform([input_list])

        # 타겟 예측
        y_RESULT = int(self.model.predict(input_scaled, verbose=0))
        # print(f'✅ y_RESULT : {y_RESULT}')
        
        return y_RESULT
        
