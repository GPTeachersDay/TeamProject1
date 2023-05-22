import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
import tensorflow as tf


class EvalAccuracy(keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
        super(EvalAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name=name, initializer="zeros")

    def update_state(self, y_true, y_predict, sample_weight=None):
        value = tf.abs((y_predict - y_true) / y_true)
        self.correct.assign(tf.reduce_mean(value))

    def result(self):
        return 1 - self.correct

    def reset_states(self):
        self.correct.assign(0.)


class Model_1():
    def __init__(self):
        pass
        # print(f'✅ 인스턴스 생성3')
        # print(f'✅ 현재 위치 : {os.getcwd()}')

    def load_data(self
                  , csv_path='Regression_data.csv'
                  , TRAIN_RATIO=0.8):
        
        # 데이터셋 로드
        df = pd.read_csv(csv_path)
        
        # 전체무게가 음수인 경우 삭제
        minus_list = df['Whole weight'] - (df['Shucked weight'] + df['Viscera weight'] + df['Shell weight'])
        minus_list = minus_list[minus_list < 0]
        df.drop(minus_list.index, axis=0, inplace=True)
        
        # 껍질의 넓이 ( a * b * π)
        df['Area'] = 0.5 * df['Length'] * 0.5 * df['Diameter'] * np.pi
            
        # 껍질의 둘레 (근사) ( 2π*(0.5 * √(a^2 + b^2)))
        df['Perimeter'] = np.pi * np.sqrt(0.5 * ((df['Length'] ** 2) + (df['Diameter'] ** 2)))
        
        # 성별 원핫 인코딩
        df=pd.get_dummies(df,columns=['Sex'])
        
        # 학습 데이터 분리
        X = df.drop('Rings', axis=1)
        y = df['Rings'].astype('float32')
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_RATIO, random_state = 83)
        
        # MinMaxScaler
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)
        
        return df, X, y, X_train, X_test, y_train, y_test
    
    def get_model(self, shape, node_n=16, dout=0.1):
        inputs  = keras.Input(shape=shape)
    
        dense1  = keras.layers.Dense(node_n)(inputs)
        norm1   = keras.layers.BatchNormalization()(dense1)
        relu1   = keras.layers.Activation('relu')(norm1)
        dout1   = keras.layers.Dropout(dout)(relu1)
        
        dense2  = keras.layers.Dense(node_n//2)(dout1)
        norm2   = keras.layers.BatchNormalization()(dense2)
        relu2   = keras.layers.Activation('relu')(norm2)
        dout2   = keras.layers.Dropout(dout)(relu2)
        
        dense3  = keras.layers.Dense(node_n//4)(dout2)
        norm3   = keras.layers.BatchNormalization()(dense3)
        relu3   = keras.layers.Activation('relu')(norm3)
        dout3   = keras.layers.Dropout(dout)(relu3)
        
        dense4  = keras.layers.Dense(node_n//8)(dout3)
        norm4   = keras.layers.BatchNormalization()(dense4)
        relu4   = keras.layers.Activation('relu')(norm4)
        dout4   = keras.layers.Dropout(dout)(relu4)
        
        dense5  = keras.layers.Dense(node_n//16)(dout4)
        norm5   = keras.layers.BatchNormalization()(dense5)
        relu5   = keras.layers.Activation('relu')(norm5)
        dout5   = keras.layers.Dropout(dout)(relu5)
        
        concat1  = keras.layers.Concatenate(axis=1)([dout5, dout4])
        dense6  = keras.layers.Dense(node_n//8)(concat1)
        norm6   = keras.layers.BatchNormalization()(dense6)
        relu6   = keras.layers.Activation('relu')(norm6)
        dout6   = keras.layers.Dropout(dout)(relu6)
        
        concat2  = keras.layers.Concatenate(axis=1)([dout6, dout3])
        dense7  = keras.layers.Dense(node_n//4)(concat2)
        norm7   = keras.layers.BatchNormalization()(dense7)
        relu7   = keras.layers.Activation('relu')(norm7)
        dout7   = keras.layers.Dropout(dout)(relu7)
        
        concat3  = keras.layers.Concatenate(axis=1)([dout7, dout2])
        dense8  = keras.layers.Dense(node_n//2)(concat3)
        norm8   = keras.layers.BatchNormalization()(dense8)
        relu8   = keras.layers.Activation('relu')(norm8)
        dout8   = keras.layers.Dropout(dout)(relu8)
        
        concat4  = keras.layers.Concatenate(axis=1)([dout8, dout1])
        dense9  = keras.layers.Dense(node_n)(concat4)
        norm9   = keras.layers.BatchNormalization()(dense9)
        relu9   = keras.layers.Activation('relu')(norm9)
        dout9   = keras.layers.Dropout(dout)(relu9)
        
        dense10  = keras.layers.Dense(node_n//2)(dout9)
        norm10   = keras.layers.BatchNormalization()(dense10)
        relu10   = keras.layers.Activation('relu')(norm10)
        dout10   = keras.layers.Dropout(dout)(relu10)
        
        outputs = keras.layers.Dense(1)(dout10)
        model   = keras.Model(inputs, outputs, name='Abalone_Model')
        
        model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MeanSquaredError(),
            metrics=[EvalAccuracy()])
        
        return model
