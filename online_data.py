import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# import train data
TRAIN_DATA_URL = "./data/train/california/online_data.csv"
TEST_DATA_URL = "./data/test/auckland/data.csv"

df = pd.read_csv(TRAIN_DATA_URL, thousands=',')
#t_df = pd.read_csv(TEST_DATA_URL, thousands=',')

df['split'] = np.random.randn(df.shape[0], 1)
msk = np.random.rand(len(df)) <= 0.7

# drop unnessesary data
df.dropna(inplace=True)
df = df.drop('ocean_proximity', axis=1)

# assign train and test data
train = df[msk]
test = df[~msk]

# split data from market valuation
X = train.drop('median_house_value', axis = 1)
y = train['median_house_value']

# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# set test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
model.fit(x = X_train, y = y_train.values, validation_data = (X_test,y_test.values), batch_size = 128, epochs = 400, callbacks = [early_stop])