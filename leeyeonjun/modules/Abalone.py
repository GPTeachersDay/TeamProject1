import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


class Modeling:
    def __init__(self):
        print(f'✅ 인스턴스 생성')
        
    def load_data(self
                  , data_path="../colabo/Data/Regression_data.csv"
                  , TRAIN_RATIO=0.8):
        
        # 데이터셋 로드
        df = pd.read_csv(data_path)
        
        # 성별 원핫 인코딩
        df = pd.get_dummies(df,columns=['Sex'])
        
        # 전체무게가 음수인 경우 삭제
        minus_list = df['Whole weight'] - (df['Shucked weight'] + df['Viscera weight'] + df['Shell weight'])
        minus_list = minus_list[minus_list < 0]
        df.drop(minus_list.index, axis=0, inplace=True)
        
        # 껍질의 넓이 ( a * b * π)
        df['Area'] = 0.5 * df['Length'] * 0.5 * df['Diameter'] * np.pi
        
        # 껍질의 둘레 (근사) ( 2π*(0.5 * √(a^2 + b^2)))
        df['Perimeter'] = np.pi * np.sqrt(0.5 * ((df['Length'] ** 2) + (df['Diameter'] ** 2)))
        
        # 학습 데이터 분리
        X = df.drop('Rings', axis=1)
        y = df['Rings']
        X_train, X_test, y_train, y_test = train_test_split(X, y
                                            , train_size=TRAIN_RATIO
                                            , random_state = 83)
        return df, X, y, X_train, X_test, y_train, y_test
    
    def get_model(self
                  , LEARNING_RATE=0.01):
        np.random.seed(42)
        tf.random.set_seed(42)
        
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu', input_shape=(len(X_train_2.keys()),)),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ])

class EvalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="eval_accuracy", **kwargs):
        super(EvalAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_predict, sample_weight=None):
        value = tf.abs((y_predict - y_true) / y_true)
        self.correct.assign(tf.reduce_mean(value))

    def result(self):
        return 1 - self.correct

    def reset_states(self):
        self.correct.assign(0.)