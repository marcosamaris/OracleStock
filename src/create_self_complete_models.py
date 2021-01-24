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

n_steps_in, n_steps_out = 21, 5
epochs = 1000
verbose=0
save = False
update = False
samples_test = 15

interval='1d'


for stock in cs.stocks_codigo[int(sys.argv[1]):int(sys.argv[1]) + len(cs.stocks_codigo)]:
    print(stock)
    
    df_target = DLmodels.get_stock_data(stock, interval)
        
        
    df_target['Moving_av']= df_target['Adj Close'].rolling(window=20,min_periods=0).mean()

    i=1
    upper_volatility=[df_target.iloc[0]['Moving_av']] 
    lower_volatility=[df_target.iloc[0]['Moving_av']] 
    while i<len(df_target):
        upper_volatility.append(df_target.iloc[i-1]['Moving_av']+3/100*df_target.iloc[i-1]['Moving_av'])
        lower_volatility.append(df_target.iloc[i-1]['Moving_av']-3/100*df_target.iloc[i-1]['Moving_av'])
        i+=1
       
    df_target['Upper_volatility']=upper_volatility
    df_target['Lower_volatility']=lower_volatility

    df_target['Short_resistance']= df_target['High'].rolling(window=10,min_periods=0).max()
    df_target['Short_support']= df_target['Low'].rolling(window=10,min_periods=0).min()
    df_target['Long_resistance']= df_target['High'].rolling(window=50,min_periods=0).max()
    df_target['Long_support']= df_target['Low'].rolling(window=50,min_periods=0).min()

    
    dataset = df_target[['Open', 'High', 'Low', 'Close', 'Volume', 'HighLoad',
       'Change', 'Moving_av', 'Upper_volatility', 'Lower_volatility',
       'Short_resistance', 'Short_support', 'Long_resistance', 'Long_support', 'Adj Close']].values
    
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)

    scaler = MinMaxScaler(feature_range=[0,1])        
    dataset = scaler.fit_transform(dataset)
    scaler_filename = 'scalers/' + stock + '_complete_' + interval + '.save'        
    pickle.dump(scaler, open(scaler_filename, 'wb'))
    
    X, y = DLmodels.split_sequences(dataset[:-samples_test], n_steps_in, n_steps_out)        
    n_features = X.shape[2]        
    y = y[:,:,-1:]
    
    stock, model = DLmodels.model_LSTM('complete_'+stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
    
    stock, model = DLmodels.model_BidirectionalLSTM('complete_'+stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)

    stock, model = DLmodels.model_convLSTM1D('complete_'+stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
        
    stock, model = DLmodels.model_ConvLSTM2D('complete_'+stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
