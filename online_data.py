'''
        This file is for american (Califonian) house data.
'''

import functools

import numpy as np
from numpy.random import RandomState
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU, Activation
    
# import train data
TRAIN_DATA_URL = "./data/train/kings_county/data.csv"
TEST_DATA_URL = "./data/test/auckland/data.csv"

# read and import the data
df = pd.read_csv(TRAIN_DATA_URL, thousands=',')
#t_df = pd.read_csv(TEST_DATA_URL, thousands=',')

# split the data by creating a test or train column
# df['split'] = np.random.randn(df.shape[0], 1)
# msk = np.random.rand(len(df)) <= 0.7

# drop unnessesary data
# df.dropna(inplace=True)
df = df.drop('date', axis=1)
df = df.drop('id', axis=1)

# assign train and test data
# rng = RandomState()
train = df #.sample(frac=0.7, random_state=rng)
# test = df.loc[~df.index.isin(train.index)]

# split data from market valuation
X = train.drop('price', axis = 1)
y = train['price']

X, y = shuffle(X, y)

# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# set test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 3)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# set the model to tensorflow sequential
model = keras.Sequential([
        Dense(20, activation=tf.nn.relu),
        Dense(1)
])

# Optimisation

model.add(Dense(126, input_dim=15)) #Dense(output_dim(also hidden wight), input_dim = input_dim)
model.add(LeakyReLU(alpha = 0.1)) #Activation

model.add(Dense(252))
model.add(Dropout(0.5))
model.add(LeakyReLU(alpha = 0.1))
model.add(Dense(1))
model.add(Activation('linear'))

adam = tf.keras.optimizers.Adam(0.0001)
# Compile Model with the adam optimizer
model.compile(loss='mean_squared_error', metrics=['mse'], optimizer = 'rmsprop')

# set early stopping to reduce overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'loss', patience = 1)

# fit the model and start epochs
model.fit(x = X_train, y = y_train.values,  epochs = 400, callbacks = [early_stop], validation_split = 0.2) #, validation_data = (X_test, y_test.values), batch_size = 128, epochs = 400, verbose=1, validation_split = 0.3, callbacks = [early_stop]

# save model
model.save('hpp.h5') 