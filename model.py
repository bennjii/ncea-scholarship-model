from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import functools
import numpy as np
from numpy.random import RandomState
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

TRAIN_DATA_URL = "./data/train/kings_county/data.csv"
TEST_DATA_URL = "./data/test/auckland/data.csv"

# read and import the data
df = pd.read_csv(TRAIN_DATA_URL, thousands=',')

df.dropna(inplace=True)
df = df.drop('date', axis=1)
df = df.drop('id', axis=1)

train = df

X = train.drop('price', axis = 1)
y = train['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = tf.keras.models.load_model('hpp.h5') 

# results = model.evaluate(X_test, y_test.values, batch_size=128) 
# print('loss and acc: ', results)

predictions = model.predict(X_test)
values = y_test.values

for i in range(0, np.size(predictions)):
        print('actual value: ', values[i])
        print('prediction:   ', predictions[i][0])
        print('\n')