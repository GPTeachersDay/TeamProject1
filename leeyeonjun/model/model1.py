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
        # print(f'✅ 인스턴스 생성1')
        # print(f'✅ 현재 위치 : {os.getcwd()}')

    def load_data(self
                  , csv_path='../Data/Regression_data.csv'
                  , TRAIN_RATIO=0.8):
        
        # 데이터셋 로드
        df = pd.read_csv(csv_path)
        
        # # 전체무게가 음수인 경우 삭제
        # minus_list = df['Whole weight'] - (df['Shucked weight'] + df['Viscera weight'] + df['Shell weight'])
        # minus_list = minus_list[minus_list < 0]
        # df.drop(minus_list.index, axis=0, inplace=True)
        
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
    
    def get_model(self, shape, layer_n=7):
        inputs  = keras.Input(shape=shape)
        x       = keras.layers.Dense(8)(inputs)
        x       = keras.layers.BatchNormalization()(x)
        x       = keras.layers.Activation('relu')(x)
        for i in range(layer_n):
            x       = keras.layers.Dense(10)(x)
            x       = keras.layers.BatchNormalization()(x)
            x       = keras.layers.Activation('relu')(x)
            x       = keras.layers.Dense(8)(x)
            x       = keras.layers.BatchNormalization()(x)
            x       = keras.layers.Activation('relu')(x)
        outputs = keras.layers.Dense(1)(x)
        model   = keras.Model(inputs, outputs, name='Abalone_Model')
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=keras.losses.MeanSquaredError(),
            metrics=[EvalAccuracy()])
        
        return model
