import functions as DLmodels
import constants_stocks as cs
import math
import sys
import os

import matplotlib.pyplot as plt
# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
# plt.rcParams.update({'figure.max_open_warning': 0})
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from matplotlib import style
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

style.use('ggplot')

n_steps_in, n_steps_out = 30, 1
epochs = 500
verbose = 0
save = True
update = True
samples_test = 10

interval='1wk'

main_df, cor =  DLmodels.get_correlation_stock_matrix(cs.stocks_codigo)
datagrouped = DLmodels.data_grouped_foreign_stock(cs.foreign_stocks, '1d')       
datagrouped = datagrouped.interpolate(method='linear', limit_direction='forward', axis=0)

for stock in cs.stocks_codigo[int(sys.argv[1]):int(sys.argv[1]) + len(cs.stocks_codigo)]:
    print(stock)    
        
    dataframe = DLmodels.get_stock_data(stock, interval) 
    if len(dataframe) < 100:
        continue

    dataframe['Moving_av']= dataframe['Adj Close'].rolling(window=20,min_periods=0).mean()

    i=1
    upper_volatility=[dataframe.iloc[0]['Moving_av']]
    lower_volatility=[dataframe.iloc[0]['Moving_av']]
    while i<len(dataframe):
        upper_volatility.append(dataframe.iloc[i-1]['Moving_av']+3/100*dataframe.iloc[i-1]['Moving_av'])
        lower_volatility.append(dataframe.iloc[i-1]['Moving_av']-3/100*dataframe.iloc[i-1]['Moving_av'])
        i+=1 
       
    dataframe['Upper_volatility']=upper_volatility
    dataframe['Lower_volatility']=lower_volatility

    dataframe['Short_resistance']= dataframe['High'].rolling(window=10,min_periods=0).max()
    dataframe['Short_support']= dataframe['Low'].rolling(window=10,min_periods=0).min()
    dataframe['Long_resistance']= dataframe['High'].rolling(window=50,min_periods=0).max()
    dataframe['Long_support']= dataframe['Low'].rolling(window=50,min_periods=0).min()
    

    #dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    
    #df_relevant = DLmodels.relevant_stocks(stock,main_df, cor)
    #df_relevant.index = pd.to_datetime(df_relevant.index) 
    #dataframe.merge(df_relevant, how='inner', on='Date')


    #dataframe = dataframe.merge(datagrouped, how='inner', on='Date')
    #dataframe = dataframe.drop(['Date'], axis=1)
    dataframe = dataframe[['Adj Close', 'Moving_av', 'Upper_volatility', 'Lower_volatility', 'Long_resistance', 'Long_support']]
    list_columns = list(dataframe.columns)
    index_adj_close = list_columns.index('Adj Close')

    dataset = dataframe.dropna().ffill().values
    
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset.reshape(-1,1))
    scaler_filename = 'scalers/Standard_week_' + stock + '-' + interval + '.save'        
    pickle.dump(scaler, open(scaler_filename, 'wb'))
    
    scaler = MinMaxScaler(feature_range=[0,1])        
    dataset = scaler.fit_transform(dataset)
    scaler_filename = 'scalers/MinMax_week_' + stock + '-' + interval + '.save'        
    pickle.dump(scaler, open(scaler_filename, 'wb'))

    X, y = DLmodels.split_sequences(dataset[:-samples_test], n_steps_in, n_steps_out)        
    n_features = X.shape[2]        
    y = y[:,:,index_adj_close:index_adj_close+1]

    DLmodels.model_LSTM(stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
    DLmodels.model_BidirectionalLSTM(stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
    DLmodels.model_convLSTM1D(stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
    DLmodels.model_ConvLSTM2D(stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
