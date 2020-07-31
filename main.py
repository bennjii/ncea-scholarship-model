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

# split data from market valuation
X = df.drop('market_value', axis = 1)
y = df['market_value']