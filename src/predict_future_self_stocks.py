import sys

import numpy as np
import pandas as pd
import constants_stocks as cs
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pickle
import functions as DLmodels

n_steps_in, n_steps_out = 7, 1

interval='1wk'
samples_test = 1
all_MAPE = []
ML_Techniques = ['LSTM', 'BidirectionalLSTM', 'convLSTM1D', 'convLSTM2D']     


for stock in cs.stocks_codigo[int(sys.argv[1]):int(sys.argv[1]) + len(cs.stocks_codigo)]:
    print(stock)
    (flag, symbol) = (True, stock)
    if flag:
        
        dataframe = DLmodels.get_stock_data(symbol, interval)
        
        dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume', 'HighLoad', 'Change', 'Adj Close']]
        
        dataframe = DLmodels.clean_dataset(dataframe)
        scaler = StandardScaler()
        dataset = scaler.fit_transform(dataframe.ffill().values)

        scaler_filename = 'scalers/' + stock + '-' + interval + '.save'
        scaler = pickle.load(open(scaler_filename, 'rb'))
        dataset = scaler.fit_transform(dataframe.iloc[:,:].ffill().values)
        
        n_features = dataset.shape[1]  
        
        X = np.reshape(dataset[-n_steps_in:, :], (1,n_steps_in,n_features))            
              
        y = dataset[-1:,-1:]
        
        for ML_tech in ML_Techniques: 
           
            if ML_tech == 'lr':
                (predictions, accuracy) = DLmodels.stock_forecasting(dataframe, n_steps_out)
            if ML_tech == 'lrLog':
                (predictions, accuracy) = DLmodels.stock_forecastingLog(dataframe, n_steps_out)                
            if ML_tech == 'conv':
                predictions = DLmodels.predict_conv1D(stock, np.reshape(X[-samples_test:], (samples_test, n_steps_in, n_features)), interval, n_steps_in)
            if ML_tech == 'LSTM':
                predictions = DLmodels.predict_LSTM(stock, np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features)), interval, n_steps_in)
            if ML_tech == 'BidirectionalLSTM':
                predictions = DLmodels.predict_BidirectionalLSTM(stock, np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features)), interval, n_steps_in)
            if ML_tech == 'convLSTM1D':
                predictions = DLmodels.predict_convLSTM1D(stock, np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features, 1)), interval, n_steps_in)
            if ML_tech == 'convLSTM2D':
                predictions = DLmodels.predict_convLSTM2D(stock, np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features, 1, 1)), interval, n_steps_in)           
            
            
            predictions = np.concatenate([np.reshape(y[-1:], (1,)), np.reshape(predictions, (n_steps_out,))], axis = 0)
       
    
            (gain_pred, index_maximo_pred) = DLmodels.predictions_max_index(predictions, n_steps_out)
        
            all_MAPE.append([stock, ML_tech, gain_pred, index_maximo_pred, n_steps_in, n_steps_out, interval, samples_test])
        
dataFrame = pd.DataFrame(all_MAPE)

dataFrame.columns = ['Stock', 'ML', 'gain_pred', 'index_maximo_pred', 'n_steps_in', 'n_steps_out', 'interval', 'samples_test']          
          
dataFrame.to_csv('logs/data_future_self.csv')

