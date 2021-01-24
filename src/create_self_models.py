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

n_steps_in, n_steps_out = 7, 1
epochs = 1000
verbose=0
save = False
update = False
samples_test = 5

interval='1wk'


for stock in cs.stocks_codigo[int(sys.argv[1]):int(sys.argv[1]) + len(cs.stocks_codigo)]:
    print(stock)
    (flag, symbol) = (True, stock)
    if flag:
        
        dataframe = DLmodels.get_stock_data(symbol, interval)        
        dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume', 'HighLoad', 'Change', 'Adj Close']]
               
        scaler = StandardScaler()
        dataset = scaler.fit_transform(dataframe.ffill().values)

        scaler = MinMaxScaler(feature_range=[0,1])        
        dataset = scaler.fit_transform(dataset)
        scaler_filename = 'scalers/' + stock + '-' + interval + '.save'        
        pickle.dump(scaler, open(scaler_filename, 'wb'))
        
        X, y = DLmodels.split_sequences(dataset[:-samples_test], n_steps_in, n_steps_out)        
        n_features = X.shape[2]        
        y = y[:,:,-1:]
        
        stock, model = DLmodels.model_LSTM(symbol, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
        
        stock, model = DLmodels.model_BidirectionalLSTM(symbol, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)

        stock, model = DLmodels.model_convLSTM1D(symbol, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
            
        stock, model = DLmodels.model_ConvLSTM2D(symbol, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
