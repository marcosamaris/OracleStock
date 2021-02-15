import functions as DLmodels
import constants_stocks as cs
import math
import sys

import matplotlib.pyplot as plt
# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
# plt.rcParams.update({'figure.max_open_warning': 0})


import numpy as np
import pandas as pd

from matplotlib import style
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

import pickle

style.use('ggplot')

n_steps_in, n_steps_out = 14, 1
epochs = 1000
verbose=0
save = True
update = True
samples_test = 5

interval='1wk'

data_trend_pt_br = pd.read_csv('./logs/trends_pt-BR.csv')
data_trend_en_us = pd.read_csv('./logs/trends_en-US.csv')
geo_us = 'en_us'
geo_br = 'pt-BR'
        
for stock in cs.stocks_codigo[int(sys.argv[1]):int(sys.argv[1]) + len(cs.stocks_codigo)]:
    print(stock)    
        
    dataframe = DLmodels.get_stock_data(stock, interval) 

    df_trend_stock_en_us = DLmodels.get_stock_trend(stock, data_trend_en_us[['date',stock]], geo_us)        
    df_trend_stock_pt_br = DLmodels.get_stock_trend(stock, data_trend_pt_br[['date',stock]], geo_br)

    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    dataframe = dataframe.merge(df_trend_stock_en_us,how="inner",on="Date")
    dataframe = dataframe.merge(df_trend_stock_pt_br,how="inner",on="Date")

    dataset = dataframe.drop(['Date'], axis=1).dropna().ffill().values

    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)

    scaler = MinMaxScaler(feature_range=[0,1])        
    dataset = scaler.fit_transform(dataset)
    scaler_filename = 'scalers/' + stock + '-' + interval + '.save'        
    pickle.dump(scaler, open(scaler_filename, 'wb'))

    X, y = DLmodels.split_sequences(dataset[:-samples_test], n_steps_in, n_steps_out)        
    n_features = X.shape[2]        
    y = y[:,:,-1:]

    DLmodels.model_LSTM(symbol, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)

    DLmodels.model_BidirectionalLSTM(symbol, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)

    DLmodels.model_convLSTM1D(symbol, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)

    DLmodels.model_ConvLSTM2D(symbol, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
