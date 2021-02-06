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
import dill 

from neupy import algorithms
from neupy.layers import *


style.use('ggplot')

n_steps_in, n_steps_out = 7, 1
epochs = 100
verbose=0
save = True
update = True
samples_test = 5

interval='1wk'
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



for stock in cs.stocks_codigo[int(sys.argv[1]):int(sys.argv[1]) + len(cs.stocks_codigo)]:
    print(stock)
        
    dataframe = get_stock_data(stock, interval)        
    dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume', 'HighLoad', 'Change', 'Adj Close']]
           
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataframe.ffill().values)

    scaler = MinMaxScaler(feature_range=[0,1])        
    dataset = scaler.fit_transform(dataset)
    scaler_filename = 'scalers/' + stock + '-' + interval + '.save'        
    pickle.dump(scaler, open(scaler_filename, 'wb'))
    
    X, y = split_sequences(dataset[:-samples_test], n_steps_in, n_steps_out)        
    n_features = X.shape[2]        
    y = y[:,:,-1:]


    x_train = X.reshape(len(X), n_steps_in*n_features)

    y_train = y.reshape(len(X), n_steps_out)

    network = Input(n_steps_in*n_features) >> Sigmoid(50) >> Sigmoid(n_steps_out)
    model = algorithms.LevenbergMarquardt(network,verbose=False,show_epoch=5)
    model.fit(x_train, y_train, epochs=epochs) 
    
    filename = 'models/' + stock + '-LevenbergMarquardt-' + interval +'.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
