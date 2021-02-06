import constants_stocks as cs
import math
import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

import pickle
from neupy import algorithms
from neupy.layers import *

def get_stock_data(stock, interval):

    df = pd.read_csv('data/' + stock  + '-' + interval + '.csv')
     
    df['HighLoad'] = (df['High'] - df['Close']) / df['Close'] * 100.0
    df['Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    return df


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

    
n_steps_in, n_steps_out = 14, 5
epochs = 100
verbose=0
save = True
update = True
samples_test = 15

interval='1d'


for stock in cs.stocks_codigo[int(sys.argv[1]):int(sys.argv[1]) + len(cs.stocks_codigo)]:
    print(stock)
    
    df_target = get_stock_data(stock, interval)
        
        
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
    
    X, y = split_sequences(dataset[-300:-samples_test], n_steps_in, n_steps_out)        
    n_features = X.shape[2]        
    y = y[:,:,-1:]


    x_train = X.reshape(len(X), n_steps_in*n_features)

    y_train = y.reshape(len(X), n_steps_out)

    network = Input(n_steps_in*n_features) >> Sigmoid(50) >> Sigmoid(n_steps_out)
    model = algorithms.LevenbergMarquardt(network,verbose=False,show_epoch=5)
    model.fit(x_train, y_train, epochs=epochs) 
    
    filename = 'models/complete_' + stock + '-LevenbergMarquardt-' + interval +'.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
