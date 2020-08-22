'''
        This file is for new zealand data
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
TRAIN_DATA_URL = "./data/test/specific/output_1.csv"
# TEST_DATA_URL = "./data/test/auckland/data.csv"

# read and import the data
df = pd.read_csv(TRAIN_DATA_URL, thousands=',', index_col=False)
#t_df = pd.read_csv(TEST_DATA_URL, thousands=',')

# split the data by creating a test or train column
# df['split'] = np.random.randn(df.shape[0], 1)
# msk = np.random.rand(len(df)) <= 0.7
# df.dropna(inplace=True)

#print(df['Market Value $'].values)

df['Market Value $'] = df['Market Value $'].str.replace('$', '')
df['Market Value $'] = df['Market Value $'].str.replace(',', '')
df['Market Value $'] = df['Market Value $'].astype('float')

# drop unnessesary and unusable data
#df = df.drop(['address', 'owners', 'suburb', 'town', 'ta_name', 'property_type', 'sale_date', 'listing_date', 'provisional_sale_price', 'provisional_sale_date', 'building_age', 'capital_value', 'rem', 'rem2'], axis=1)
df = df.drop(["Job Code", "Valuation Date", "Existing/New", "Lot", "Street No.", "Street Name", "Locality", "Type", "Land Value $", "New Rate $", "Outdoor Areas", "OIs", "OBs", "Chattels $", "Rent", "Comments"], axis=1)

# assign train and test data
# rng = RandomState()


#.sample(frac=0.7, random_state=rng)
# test = df.loc[~df.index.isin(train.index)]
df = df[df['Land Area'] != 'Unit Title']
df = df[df['Land Area'] != 'Cross Lease']

df['Beds'] = df['Beds'].astype('float')
df['Land Area'] = df['Land Area'].astype('float')
print(df.dtypes)

#df = np.asarray(df)
#df = df.asarray(X).astype(np.float32)

train = df 

# split data from market valuation
X = train.drop('Market Value $', axis = 1)
y = train['Market Value $']

# print(X)
# print(y)

X, y = shuffle(X, y)
tf.keras.backend.set_floatx('float64')

# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# set test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)
# scaler = MinMaxScaler()

# print(X_train)
# X_train = scaler.fit_transform(X_train)
# X_test  = scaler.transform(X_test)
# print(X_train)

# set the model to tensorflow sequential
model = keras.Sequential([
        Dense(20, activation=tf.nn.relu),
        Dense(5)
])

# Optimisation

model.add(Dense(126, input_dim=15)) #Dense(output_dim(also hidden wight), input_dim = input_dim)
model.add(LeakyReLU(alpha = 0.1)) #Activation

# model.add(Dense(252))
model.add(Dropout(0.5))
model.add(LeakyReLU(alpha = 0.1))
model.add(Dense(1))
model.add(Activation('linear'))

adam = tf.keras.optimizers.Adam(0.0001)
# Compile Model with the adam optimizer
model.compile(loss='mean_squared_error', metrics=['mse'], optimizer = 'rmsprop')

# set early stopping to reduce overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'loss', patience = 5)

# fit the model and start epochs
model.fit(x = X_train, y = y_train.values,  epochs = 400, callbacks = [early_stop], validation_split = 0.2, batch_size = 64) #, validation_data = (X_test, y_test.values), batch_size = 128, epochs = 400, verbose=1, validation_split = 0.3, callbacks = [early_stop]

# save model
model.save('hpp.h5') 