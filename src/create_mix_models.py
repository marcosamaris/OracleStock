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

n_steps_in, n_steps_out = 3, 3
samples_test = 5
epochs = 1000
verbose=0
save = True
update = True

interval='1d'




datagrouped = DLmodels.data_grouped_foreign_stock(cs.foreign_stocks, interval)
all_result = []

for stock in cs.stocks_codigo[int(sys.argv[1]):int(sys.argv[1]) + len(cs.stocks)]:
    
    (flag, symbol) = (True, stock)
    if flag:

        dataframe = DLmodels.get_stock_data(symbol, interval)

        df2 = dataframe[['Volume', 'Adj Close']]        
        df2.index = dataframe['Date']
        dataframe = pd.merge(datagrouped,df2, how='inner', left_index=True, right_index=True)

        dataframe = DLmodels.clean_dataset(dataframe)
        

        scaler = StandardScaler()
        dataset = dataframe.values
        dataset = scaler.fit_transform(dataset)        

        scaler = MinMaxScaler(feature_range=[0,1])        
        dataset = scaler.fit_transform(dataset)
        scaler_filename = 'scalers/' + stock + '-' + interval + '-FG.save'
        pickle.dump(scaler, open(scaler_filename, 'wb'))

        print(stock + ': ' + str(len(dataset)))
        X, y = DLmodels.split_sequences(dataset[:-samples_test], n_steps_in, n_steps_out)
        print(stock + ': ' + str(X.shape))

        X = X[:, :, :]
        shape_x = X.shape

        y = y[:,:,-1:]

        # stock, model = DLmodels.model_conv1D('FG-' + symbol, X, y, interval, n_steps_in, n_steps_out,epochs, save, update, verbose)

        stock, model = DLmodels.model_LSTM('FG-' + symbol, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
        stock, model = DLmodels.model_BidirectionalLSTM('FG-' + symbol, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)

        stock, model = DLmodels.model_convLSTM1D('FG-' + symbol, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)

        stock, model = DLmodels.model_ConvLSTM2D('FG-' + symbol, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
