import functions as DLmodels
import constants_stocks as cs
import math
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


n_steps_in, n_steps_out = 50, 3
epochs = 1000
verbose=0
save = True
update = True
samples_test = 15

interval='1d'
data_trend_pt_br = pd.read_csv('./logs/trends_pt-BR.csv')
data_trend_en_us = pd.read_csv('./logs/trends_en-US.csv')
geo_us = 'en_us'
geo_br = 'pt-BR'
main_df, cor =  DLmodels.get_correlation_stock_matrix(cs.stocks_codigo)
        
for stock in cs.stocks_codigo[int(sys.argv[1]):int(sys.argv[1]) + len(cs.stocks_codigo)]:
    print(stock)
    
    dataframe = DLmodels.get_stock_data(stock, interval)        
    if len(dataframe) < 200:
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
    
    
    df_trend_stock_en_us = DLmodels.get_stock_trend(stock, data_trend_en_us[['date',stock]], geo_us)        
    df_trend_stock_pt_br = DLmodels.get_stock_trend(stock, data_trend_pt_br[['date',stock]], geo_br)

    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    #dataframe = dataframe.merge(df_trend_stock_en_us,how="inner",on="Date")
    #dataframe = dataframe.merge(df_trend_stock_pt_br,how="inner",on="Date")

    df_relevant = DLmodels.relevant_stocks(stock,main_df, cor)
    df_relevant.index = pd.to_datetime(df_relevant.index)
    dataframe = dataframe.merge(df_relevant, how='inner', on='Date')

    dataset = DLmodels.clean_dataset(dataframe.drop(['Date'], axis=1)).values
    
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)

    scaler = MinMaxScaler(feature_range=[0,1])        
    dataset = scaler.fit_transform(dataset)
    scaler_filename = 'scalers/complete_' + stock + '_' + interval + '.save'        
    pickle.dump(scaler, open(scaler_filename, 'wb'))
    
    X, y = DLmodels.split_sequences(dataset[:-samples_test], n_steps_in, n_steps_out)        
    n_features = X.shape[2]        
    y = y[:,:,-1:]
    
    DLmodels.model_LSTM('complete_'+stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
    
    DLmodels.model_BidirectionalLSTM('complete_'+stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)

    DLmodels.model_convLSTM1D('complete_'+stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
        
    DLmodels.model_ConvLSTM2D('complete_'+stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose)
