import sys

import numpy as np
import pandas as pd
import constants_stocks as cs
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pickle
import functions as DLmodels

n_steps_in, n_steps_out = 21, 5

samples_test = 15

ML_Techniques = ['LSTM', 'BidirectionalLSTM', 'convLSTM1D', 'convLSTM2D']     
interval='1d'
data_trend_pt_br = pd.read_csv('./logs/trends_pt-BR.csv')
data_trend_en_us = pd.read_csv('./logs/trends_en-US.csv')
geo_us = 'en_us'
geo_br = 'pt-BR'

datagrouped = DLmodels.data_grouped_foreign_stock(cs.foreign_stocks, interval)

all_MAPE = []
for stock in cs.stocks_codigo[int(sys.argv[1]):int(sys.argv[1]) + 65]:
    print(stock)
    (flag, symbol) = (True, stock)
    if flag:
        
        dataframe = DLmodels.get_stock_data(symbol, interval)
        
        df2 = dataframe        
        df2['Date'] = pd.to_datetime(df2['Date'])
        dataframe = df2.merge(datagrouped, how='inner', on='Date')
        
        df_trend_stock_en_us = DLmodels.get_stock_trend(stock, data_trend_en_us[['date',stock]], geo_us)
        df_trend_stock_pt_br = DLmodels.get_stock_trend(stock, data_trend_pt_br[['date',stock]], geo_br)

        dataframe = dataframe.merge(df_trend_stock_en_us,how="inner",on="Date")
        dataframe = dataframe.merge(df_trend_stock_pt_br,how="inner",on="Date")

        dataset = dataframe.drop(['Date'], axis=1).dropna().ffill().values        
        scaler = StandardScaler()
        dataset = scaler.fit_transform(dataset)

        scaler_filename = 'scalers/FG_' + stock + '-' + interval + '-FG.save'
        scaler = pickle.load(open(scaler_filename, 'rb'))
        dataset = scaler.fit_transform(dataset)
        
        X, y = DLmodels.split_sequences(dataset, n_steps_in, n_steps_out)
        X = X[:, :, :]
        
        n_features = X.shape[2]        
        y = y[:,:,-1:]

        
        for ML_tech in ML_Techniques: 
           
            if ML_tech == 'lr':
                (predictions, accuracy) = DLmodels.stock_forecasting(dataframe, n_steps_out)
            if ML_tech == 'lrLog':
                (predictions, accuracy) = DLmodels.stock_forecastingLog(dataframe, n_steps_out)                
            if ML_tech == 'conv':
                predictions = DLmodels.predict_conv1D(stock, np.reshape(X[-samples_test:], (samples_test, n_steps_in, n_features)), interval, n_steps_in)
            if ML_tech == 'LSTM':
                predictions = DLmodels.predict_LSTM('FG-' + stock, np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features)), interval, n_steps_in)
            if ML_tech == 'BidirectionalLSTM':
                predictions = DLmodels.predict_BidirectionalLSTM('FG-' + stock, np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features)), interval, n_steps_in)
            if ML_tech == 'convLSTM1D':
                predictions = DLmodels.predict_convLSTM1D('FG-' + stock, np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features, 1)), interval, n_steps_in)
            if ML_tech == 'convLSTM2D':
                predictions = DLmodels.predict_convLSTM2D('FG-' + stock, np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features, 1, 1)), interval, n_steps_in)   
        
            
            MAPE = DLmodels.mean_absolute_percentage_error(np.reshape(y[-samples_test:], (samples_test*n_steps_out,)), np.reshape(predictions, (samples_test*n_steps_out,)))
            
            
            all_MAPE.append([stock, ML_tech, np.around(MAPE,1), n_steps_in, n_steps_out, interval, samples_test])
       

dados = pd.DataFrame(all_MAPE)

dados.columns = ['Stock', 'ML', 'MAPE', 'n_steps_in', 'n_steps_out', 'interval', 'samples_test']          
          
dados.to_csv('./logs/dados_MAPE_mix.csv')

