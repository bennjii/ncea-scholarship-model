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

TRAIN_DATA_URL = "./data/test/specific/output_1.csv"
# TEST_DATA_URL = "./data/test/auckland/data.csv"

def difference(a, b):
        return round((a - b) / ((a + b) / 2) * 100, 2)

# read and import the data
df = pd.read_csv(TRAIN_DATA_URL, thousands=',', header = 0)

#df.dropna(inplace=True)
#@df = df.fillna(0)

#df = df.drop(['address', 'owners', 'suburb', 'town', 'ta_name', 'property_type', 'sale_date', 'listing_date', 'provisional_sale_price', 'provisional_sale_date', 'building_age', 'capital_value', 'rem', 'rem2'], axis=1)
df = df.drop(["Job Code", "Valuation Date", "Existing/New", "Lot", "Street No.", "Street Name", "Locality", "Type", "Land Value $", "New Rate $", "Outdoor Areas", "OIs", "OBs", "Chattels $", "Rent", "Comments"], axis=1)
df = df[df['Land Area'] != 'Unit Title']
df = df[df['Land Area'] != 'Cross Lease']

df['Beds'] = df['Beds'].astype('float')
df['Land Area'] = df['Land Area'].astype('float')

df['Market Value $'] = df['Market Value $'].str.replace('$', '')
df['Market Value $'] = df['Market Value $'].str.replace(',', '')
df['Market Value $'] = df['Market Value $'].astype('float')

train = df

X = train.drop('Market Value $', axis = 1)
y = train['Market Value $']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)
#scaler = MinMaxScaler()

X_save = X_test

#X_train = scaler.fit_transform(X_train)
#X_test  = scaler.transform(X_test)

model = tf.keras.models.load_model('hpp.h5') 

# results = model.evaluate(X_test, y_test.values, batch_size=128) 
# print('loss and acc: ', results)

predictions = model.predict(X_test)
values = y_test.values

#print(predictions)

# output
for i in range(4, 5):
        print('\n')
        print('\n')

        print('----[ PREDICTION ]----')
        print('\n')

        print('actual value: ', values[i])
        print('prediction:   ', predictions[i][0])
        print('\n')
        print('difference:   ', difference(values[i], predictions[i][0]), "%")

        print('\n')
        print(X_save.iloc[[i]])
        print('\n')