'''
        This file is for local (New Zealand Data)
'''

import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# import train data
TRAIN_DATA_URL = "./data/train/data.csv"
TEST_DATA_URL = "./data/test/data.csv"

df = pd.read_csv(TRAIN_DATA_URL, thousands=',')
t_df = pd.read_csv(TEST_DATA_URL, thousands=',')

def stringToInt(name):
    df[name] = [x[1:] for x in df.market_value]
    df[name] = df[name].str.replace(',', '')
    df[name] = pd.to_numeric(df[name])

# convert the market values from strings into integers
stringToInt('market_value')

# print head of data to confirm previous code work
print(df.head())

# dropping unnessesary columns
df = df.drop(['Job Code', 'Valuation Date', 'Existing/New', 'Lot', 'Street Name', 'Street No.', 'Locality', 'Type', 'Land Value $', 'Net Rate $', 'Outdoor Areas', 'OIs', 'OBs', 'Chattels $', 'Rent $', 'Comments'], axis=1)
df = df.apply(pd.to_numeric, errors='coerce')

# split data from market valuation
X = df.drop('market_value', axis = 1)
y = df['market_value']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.fit(x=X_train, y=y_train.values, validation_data=(X_test,y_test.values), batch_size=128, epochs=400, callbacks=[early_stop])