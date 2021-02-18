import sys

import numpy as np
import pandas as pd
import constants_stocks as cs
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pickle
import functions as DLmodels

n_steps_in, n_steps_out = 21, 5
samples_test = 15
interval='1d'


all_MAPE = []
ML_Techniques = ['LSTM', 'BidirectionalLSTM', 'convLSTM1D', 'convLSTM2D']     

data_trend_pt_br = pd.read_csv('./logs/trends_pt-BR.csv')
data_trend_en_us = pd.read_csv('./logs/trends_en-US.csv')
geo_us = 'en_us'
geo_br = 'pt-BR'

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

    df_trend_stock_en_us = DLmodels.get_stock_trend(stock, data_trend_en_us[['date',stock]], geo_us)
    df_trend_stock_pt_br = DLmodels.get_stock_trend(stock, data_trend_pt_br[['date',stock]], geo_br)

    df_target['Date'] = pd.to_datetime(df_target['Date'])
    dataframe = df_target
    dataframe = dataframe.merge(df_trend_stock_en_us,how="inner",on="Date")
    dataframe = dataframe.merge(df_trend_stock_pt_br,how="inner",on="Date")

    dataset = dataframe.drop(['Date'], axis=1).dropna().ffill().values
    
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)

    scaler_filename = 'scalers/complete_' + stock + '_complete_' + interval + '.save'   
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
            predictions = DLmodels.predict_conv1D(stock, 
                                                  np.reshape(X[-samples_test:],(samples_test, n_steps_in, n_features)),
                                                  interval, n_steps_in)
        if ML_tech == 'LSTM':
            predictions = DLmodels.predict_LSTM('complete_'+stock, 
                                                np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features)),
                                                interval, n_steps_in)
        if ML_tech == 'BidirectionalLSTM':
            predictions = DLmodels.predict_BidirectionalLSTM('complete_'+stock, 
                                                             np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features)), 
                                                             interval, n_steps_in)
        if ML_tech == 'convLSTM1D':
            predictions = DLmodels.predict_convLSTM1D('complete_'+stock, 
                                                      np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features, 1)), 
                                                      interval, n_steps_in)
        if ML_tech == 'convLSTM2D':
            predictions = DLmodels.predict_convLSTM2D('complete_'+stock, 
                                                      np.reshape(X[-samples_test:], (samples_test,n_steps_in, n_features, 1, 1)), 
                                                      interval, n_steps_in)   
    
        print(y[-samples_test:].shape)
        MAPE = DLmodels.mean_absolute_percentage_error(np.reshape(y[-samples_test:], (samples_test*n_steps_out,)), np.reshape(predictions, (samples_test*n_steps_out,)))
        
        all_MAPE.append([stock, ML_tech, np.around(MAPE,1), n_steps_in, n_steps_out, interval, samples_test])


dados = pd.DataFrame(all_MAPE)

dados.columns = ['Stock', 'ML', 'MAPE', 'n_steps_in', 'n_steps_out', 'interval', 'samples_test']          
          
dados.to_csv('./logs/dados_MAPE_self_complete.csv')

