import sys

import numpy as np
import pandas as pd
import constants_stocks as cs
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pickle
import functions as DLmodels

n_steps_in, n_steps_out = 21, 5
samples_test = 1
interval='1d'


all_MAPE = []
ML_Techniques = ['LSTM', 'BidirectionalLSTM', 'convLSTM1D', 'convLSTM2D']     


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
    

    scaler_filename = 'scalers/' + stock + '_complete_' + interval + '.save'   
    scaler = pickle.load(open(scaler_filename, 'rb'))
    dataset = scaler.fit_transform(dataset)
    
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
            predictions = DLmodels.predict_LSTM('complete_'+stock, np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features)), interval, n_steps_in)
        if ML_tech == 'BidirectionalLSTM':
            predictions = DLmodels.predict_BidirectionalLSTM('complete_complete_'+stock, np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features)), interval, n_steps_in)
        if ML_tech == 'convLSTM1D':
            predictions = DLmodels.predict_convLSTM1D('complete_complete_complete_'+stock, np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features, 1)), interval, n_steps_in)
        if ML_tech == 'convLSTM2D':
            predictions = DLmodels.predict_convLSTM2D('complete_complete_complete_complete_'+stock, np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features, 1, 1)), interval, n_steps_in)           
        
        
        predictions = np.concatenate([np.reshape(y[-1:], (1,)), np.reshape(predictions, (n_steps_out,))], axis = 0)
   

        (gain_pred, index_maximo_pred) = DLmodels.predictions_max_index(predictions, n_steps_out)
    
        all_MAPE.append([stock, ML_tech, gain_pred, index_maximo_pred, n_steps_in, n_steps_out, interval, samples_test])
        
dataFrame = pd.DataFrame(all_MAPE)

dataFrame.columns = ['Stock', 'ML', 'gain_pred', 'index_maximo_pred', 'n_steps_in', 'n_steps_out', 'interval', 'samples_test']          
          
dataFrame.to_csv('logs/data_future_self_complete.csv')

