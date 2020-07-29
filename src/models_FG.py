import pickle
import sys 
import time
import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from pandas_datareader import data as pdr
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters(False)
plt.rcParams.update({'figure.max_open_warning': 0})

register_matplotlib_converters()

# import keras
from tensorflow import keras

from keras import backend
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import Conv1D, Dense, LSTM, RepeatVector, TimeDistributed, ConvLSTM2D, \
                        BatchNormalization, Dropout, MaxPooling2D, UpSampling1D
from keras.callbacks import EarlyStopping



from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from multiprocessing import Pool


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


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


def load_foreign_stocks(foreign_stocks, feature = 'Close'):
    df_foreign = pd.DataFrame()

    for stock in foreign_stocks:
        temp_df = pd.read_csv('./data/' + stock + '.csv')
        temp_df.index = temp_df['Date']
    
        df_foreign = pd.concat([df_foreign, temp_df[feature]], axis=1, sort=True)
        df_foreign = df_foreign.dropna() 
    df_foreign.columns = foreign_stocks                                            

    return df_foreign
    
from sklearn.ensemble import RandomForestRegressor

def model_conv1_FG(stock):
    
    data = pd.read_csv('data/' + stock + '.csv')
    data.index = data['Date']
    data = data[data.index >= '2015-05-15']

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.loc[:,'HighLoad'] = (data['High'] - data['Close']) / data['Close'] * 100.0
    data.loc[:,'Change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0
    
    df_foreign = load_foreign_stocks(foreign_stocks) 

    data = pd.concat([data[['HighLoad', 'Change', 'Volume', 'Low', 'High', 'Close']]], axis=1, sort=True)[:-10]
    data = data.dropna()
    print(np.shape(data))
    
    scaler = MinMaxScaler(feature_range=[0,1])
    
    # define input sequence
    scaler_norm = StandardScaler()
    # define input sequence
    dataset = scaler.fit_transform(data.ffill().values)

    # choose a number of time steps
    n_steps_in, n_steps_out = 5, 3
    
    # convert into input/output
    X, y = split_sequences(dataset, n_steps_in, n_steps_out)
    
    # flatten output
    y = y[:,:,-1:]
    
    n_output = y.shape[1] * y.shape[2]
    y = y.reshape((y.shape[0], n_output))
    
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(Dropout(0.5))
    
    
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    
    # model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    # model.add(Dropout(0.5))
        
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(n_output)) 

    model.compile(optimizer='adam', loss='mse')
    
    # fit model
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X, y, epochs=200, verbose=0,     validation_split=0.05,  callbacks=[early_stop])

    filename = 'models/' + stock + '-FG-CNN-2y.h5'
    
    model.save(filename)

    del model

    keras.backend.clear_session()
    
    return (stock, True)

def model_LSTM_FG(stock):
    
    data = pd.read_csv('data/' + stock + '.csv')
    data.index = data['Date']
    data = data[data.index >= '2015-05-15']

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.loc[:,'HighLoad'] = (data['High'] - data['Close']) / data['Close'] * 100.0
    data.loc[:,'Change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0

    


    df_foreign = load_foreign_stocks(foreign_stocks) 
    
    data = pd.concat([data[['HighLoad', 'Change', 'Volume', 'Low', 'High', 'Close']]], axis=1, sort=True)[:-10]
    data = data.dropna()
    
    scaler = MinMaxScaler(feature_range=[0,1])
    # define input sequence
    scaler_norm = StandardScaler()
    # define input sequence
    dataset = scaler.fit_transform(data.ffill().values)

    # choose a number of time steps
    n_steps_in, n_steps_out = 5, 3
    # convert into input/output
    X, y = split_sequences(dataset, n_steps_in, n_steps_out)
                           
    # flatten output
    y = y[:,:,-1:]
    
    n_output = y.shape[1] * y.shape[2]
    n_features_in = X.shape[2]
    n_features_out = y.shape[2]
    
    # define model
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(n_steps_in, n_features_in)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(64, activation='relu'))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(64, activation='relu'))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(64, activation='relu'))
    
    model.add(RepeatVector(n_steps_out))

    
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(30, activation='relu')))
    model.add(TimeDistributed(Dense(20, activation='relu')))
    model.add(TimeDistributed(Dense(10, activation='relu')))
    model.add(TimeDistributed(Dense(n_features_out)))
    model.compile(optimizer='adam', loss='mse')
    # print(model.summary())
    # fit model
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X, y, epochs=200, verbose=0,    validation_split=0.05,  callbacks=[early_stop])
    
    model.save('models/' + stock + '-FG-LSTM-2y.h5')    

    del model
    keras.backend.clear_session()
    
    return (stock, True)


def model_CNN_LSTM_2D_FG(stock):

    data = pd.read_csv('data/' + stock + '.csv')
    data.index = data['Date']
    data = data[data.index >= '2015-05-15']

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.loc[:,'HighLoad'] = (data['High'] - data['Close']) / data['Close'] * 100.0
    data.loc[:,'Change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0

    


    df_foreign = load_foreign_stocks(foreign_stocks) 

    data = pd.concat([data[['HighLoad', 'Change', 'Volume', 'Low', 'High', 'Close']]], axis=1, sort=True)[:-10]

    data = data.dropna()
    
    scaler = MinMaxScaler(feature_range=[0,1])
    # define input sequence
    scaler_norm = StandardScaler()
    # define input sequence
    dataset = scaler.fit_transform(data.ffill().values)

    # choose a number of time steps
    n_steps_in, n_steps_out = 5, 3
    # convert into input/output
    X, y_complete = split_sequences(dataset, n_steps_in, n_steps_out)

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = y_complete[:,:,-1:]

    # flatten output
    n_output = y.shape[1] * y.shape[2]

    y = y.reshape((y.shape[0], n_output))

    n_features_in = X.shape[2]
    
    # define model
    model = Sequential()

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), 
                              input_shape=(n_steps_in, n_features_in, 1)))
    
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu')))                            
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu')))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu')))
    
    # model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    # model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu')))
    # model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(30, activation='relu'))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(20, activation='relu'))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(10, activation='relu'))
    
    model.add(Dense(n_output))

    model.compile(optimizer='adam', loss='mse')
    # print(model.summary())
    
    # fit model
    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(X, y, epochs=100, verbose=0,    validation_split=0.05,  callbacks=[early_stop])
    
    model.save('models/' + stock + '-FG-CNN-LSTM-2y.h5')

    del model
    keras.backend.clear_session()
        
    return (stock, True)


def model_ConvLSTM2D_FG(stock):

    data = pd.read_csv('data/' + stock + '.csv')
    data.index = data['Date']
    data = data[data.index >= '2015-05-15']

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.loc[:,'HighLoad'] = (data['High'] - data['Close']) / data['Close'] * 100.0
    data.loc[:,'Change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0

    

    df_foreign = load_foreign_stocks(foreign_stocks) 

    data = pd.concat([data[['HighLoad', 'Change', 'Volume', 'Low', 'High', 'Close']]], axis=1, sort=True)[:-10]

    data = data.dropna()

    scaler = MinMaxScaler(feature_range=[0,1])
    
    # define input sequence
    scaler_norm = StandardScaler()
    # define input sequence
    dataset = scaler.fit_transform(data.ffill().values)
	
    # choose a number of time steps
    n_steps_in, n_steps_out = 5, 3
    # convert into input/output
    X, y_complete = split_sequences(dataset, n_steps_in, n_steps_out)

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1, 1)
    y = y_complete[:,:,-1:]

    # flatten output
    n_output = y.shape[1] * y.shape[2]

    y = y.reshape((y.shape[0], n_output))

    n_features_in = X.shape[2]
    n_features_out = y.shape[1]

    # define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(2,1), activation='relu', 
                         input_shape=(n_steps_in, n_features_in, 1, 1),
                         padding = 'same', return_sequences = True))
                         
    model.add(Dropout(0.2, name = 'dropout_1')) 
    model.add(BatchNormalization(name = 'batch_norm_1'))
    model.add(ConvLSTM2D(filters=64, kernel_size=(3,1), activation='relu'))

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=1)))

    model.add(Flatten())

    #model.add(TimeDistributed(Dense(units=1, name = 'dense_1', activation = 'relu')))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    
    model.add(Dense(n_features_out))

    model.compile(optimizer='adam', loss='mse')
    # print(model.summary())
    # fit model
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    
    model.fit(X, y, epochs=100, verbose=0,    validation_split=0.05,  callbacks=[early_stop])
    
    model.save('models/' + stock + '-FG-ConvLSTM2D-2y.h5')
    keras.backend.clear_session()
    
    del model
    return (stock, True)


def plot_model_predictions(stock):
        
    plt.style.use(['ggplot'])
        
    data = pd.read_csv('data/' + stock + '.csv')
    data.index = data['Date']

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.loc[:,'HighLoad'] = (data['High'] - data['Close']) / data['Close'] * 100.0
    data.loc[:,'Change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0

    



    df_foreign = load_foreign_stocks(foreign_stocks)
    data = pd.concat([data[['HighLoad', 'Change', 'Volume', 'Low', 'High', 'Close']]], axis=1, sort=True)[:]

    data = data.dropna()
    scaler = MinMaxScaler(feature_range=[0,1])
    # define input sequence
    scaler_norm = StandardScaler()
    # define input sequence
    dataset = scaler.fit_transform(data.ffill().values)


    N_samples = 10

    # choose a number of time steps
    n_steps_in, n_steps_out = 5, 3

    # convert into input/output
    X, y_complete = split_sequences(dataset, n_steps_in, n_steps_out)

    y = y_complete[:,:,-1:]

    # the dataset knows the number of features, e.g. 2
    n_features_in = X.shape[2]
    n_features_out = 1


    names = ['Close']
    X_unnorm = scaler.inverse_transform(X[-1])
    X_test = pd.DataFrame(X_unnorm[:, -n_features_out:],columns=names)    
    X_test['Date'] = data.index[-(n_steps_in+n_steps_out):-n_steps_out]


    y_test = np.reshape(scaler.inverse_transform(y_complete[-1])[:,-n_features_out:], 
    (n_steps_out,n_features_out))
    y_test = pd.DataFrame(y_test, columns=names)
    y_test['Date'] = data.index[-n_steps_out:]    

    date_end = pd.to_datetime(data.index[-1])
    days= n_steps_out + 2
    date_begin = date_end + timedelta(days = days)
    test_date = pd.DataFrame(pd.date_range(date_end, date_begin), columns=['Date'])
    test_date['weekday'] = test_date['Date'].dt.strftime('%w').values.astype(int)
    test_date = pd.to_datetime(test_date['Date'].loc[(test_date['weekday'] != 6) & 
                                                     (test_date['weekday'] != 0)].reset_index(drop=True).values)


    plt.clf()
    plt.cla()


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True,sharey=False)

    model = load_model('models/' + stock + '-FG-CNN-2y.h5' )
    yhat = model.predict(X[-N_samples:], verbose=0)

    # print(np.reshape(y[-1], (n_steps_out*n_features_out)), np.reshape(yhat[-1], ( n_steps_out*n_features_out)))

    MAE_CNN = mean_absolute_percentage_error(np.reshape(y[-N_samples:], (N_samples*n_steps_out*n_features_out)),
                                             np.reshape(yhat[-N_samples:], (N_samples*n_steps_out*n_features_out)))

    y_pred = np.reshape(yhat[-1], (n_steps_out,n_features_out))
    y_pred = pd.DataFrame(scaler.inverse_transform(np.concatenate([y_complete[-1][:,:-n_features_out], y_pred],
     axis=1))[:,-n_features_out:],columns=names)
    y_pred['Date'] = data.index[-n_steps_out:]


    y_future = model.predict(np.reshape(dataset[-n_steps_in:], (1, n_steps_in,n_features_in)))
    y_future = np.reshape(y_future[-1], (n_steps_out,n_features_out))
    y_future = pd.DataFrame(scaler.inverse_transform(np.concatenate([y_complete[-1][:,:-n_features_out], y_future], axis=1))[:,-n_features_out:],columns=names)
    y_future['Date'] = test_date[1:]


    df_true = pd.concat([X_test[-5:],y_test],sort=False)
    df_pred = pd.concat([y_pred,y_future],sort=False)

    df_true['Date'] = pd.to_datetime(df_true['Date'])
    df_pred['Date'] = pd.to_datetime(df_pred['Date'])    


    df_true['Date'] = np.array(pd.Series(df_true['Date']).apply(lambda x: x.day).astype(str))
    df_pred['Date'] = np.array(pd.Series(df_pred['Date']).apply(lambda x: x.day).astype(str))



    fig.suptitle(stock +  ' daily prediction. Solid lines = Predicted')

    # ax1.plot( df_true['Date'], df_true['High'], linewidth=2, linestyle='--',color='blue')
    # ax1.plot(df_true['Date'], df_true['Open'], linewidth=2, linestyle='--',color='yellow')
    ax1.plot(df_true['Date'], df_true['Close'], markerfacecolor='green',
         markeredgecolor='green', linewidth=2, marker='.', linestyle='--',color='yellow', label='Real Adj Close')
    # ax1.plot(df_true['Date'], df_true['Low'], linewidth=2, linestyle='--',color='red')


    # ax1.plot(df_pred['Date'], df_pred['High'], linewidth=2,marker='^', color='blue', label='High')
    # ax1.plot(df_pred['Date'], df_pred['Open'], linewidth=2, marker='*', color='yellow', label='Open')
    ax1.plot(df_pred['Date'], df_pred['Close'], markerfacecolor='white',
         markeredgecolor='yellow', linewidth=2, marker='*', color='green', label='Predited Adj Close')
    # ax1.plot(df_pred['Date'], df_pred['Low'], linewidth=2, marker='v', color='red', label='Low')

    ax1.set_title('CNN - MAE: ' + str(round(MAE_CNN, 1)))  
    # ax1.set_xlabel('Days')
    ax1.set_ylabel('Stock Value')   

    ax1.tick_params(labelrotation=45)
    #ax1.legend()
    #ax1.grid()

    model = load_model('models/' + stock + '-FG-LSTM-2y.h5' )
    yhat = model.predict(X[-N_samples:], verbose=0)
    MAE_LSTM = mean_absolute_percentage_error(np.reshape(y[-N_samples:], (N_samples*n_steps_out*n_features_out)),
                                             np.reshape(yhat[-N_samples:], (N_samples*n_steps_out*n_features_out)))

    y_pred = np.reshape(yhat[-1], (n_steps_out,n_features_out))
    y_pred = pd.DataFrame(scaler.inverse_transform(np.concatenate([y_complete[-1][:,:-n_features_out], y_pred],
     axis=1))[:,-n_features_out:],columns=names)

    y_pred['Date'] = data.index[-n_steps_out:]

    y_future = model.predict(np.reshape(dataset[-n_steps_in:], (1, n_steps_in,n_features_in)))
    y_future = np.reshape(y_future[-1], (n_steps_out,n_features_out))
    y_future = pd.DataFrame(scaler.inverse_transform(np.concatenate([y_complete[-1][:,:-n_features_out], 
    y_future], axis=1))[:,-n_features_out:],columns=names)
    y_future['Date'] = test_date[1:]

    df_true = pd.concat([X_test[-5:],y_test],sort=False)
    df_pred = pd.concat([y_pred,y_future],sort=False)

    df_true['Date'] = pd.to_datetime(df_true['Date'])
    df_pred['Date'] = pd.to_datetime(df_pred['Date'])    

    df_true['Date'] = np.array(pd.Series(df_true['Date']).apply(lambda x: x.day).astype(str))
    df_pred['Date'] = np.array(pd.Series(df_pred['Date']).apply(lambda x: x.day).astype(str))

    # ax1.plot( df_true['Date'], df_true['High'], linewidth=2, linestyle='--',color='blue')
    # ax1.plot(df_true['Date'], df_true['Open'], linewidth=2, linestyle='--',color='yellow')
    ax2.plot(df_true['Date'], df_true['Close'], markerfacecolor='green',
         markeredgecolor='green', linewidth=2, marker='.', linestyle='--',color='yellow', label='Real Adj Close')
    # ax1.plot(df_true['Date'], df_true['Low'], linewidth=2, linestyle='--',color='red')


    # ax1.plot(df_pred['Date'], df_pred['High'], linewidth=2,marker='^', color='blue', label='High')
    # ax1.plot(df_pred['Date'], df_pred['Open'], linewidth=2, marker='*', color='yellow', label='Open')
    ax2.plot(df_pred['Date'], df_pred['Close'], markerfacecolor='white',
         markeredgecolor='yellow', linewidth=2, marker='*', color='green', label='Predited Adj Close')
    # ax1.plot(df_pred['Date'], df_pred['Low'], linewidth=2, marker='v', color='red', label='Low')

    ax2.set_title('LSTM - MAE: ' + str(round(MAE_LSTM, 1)))
    # ax2.set_xlabel('Days')
    ax2.set_ylabel('Stock Value')   

    ax2.tick_params(labelrotation=45)
    #ax2.legend()
    #ax2.grid()

    model = load_model('models/' + stock + '-FG-CNN-LSTM-2y.h5' )
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    yhat = model.predict(X[-N_samples:], verbose=0)
    MAE_CNN_LSTM = mean_absolute_percentage_error(np.reshape(y[-N_samples:], (N_samples*n_steps_out*n_features_out)),
                                             np.reshape(yhat[-N_samples:], (N_samples*n_steps_out*n_features_out)))

    y_pred = np.reshape(yhat[-1], (n_steps_out,n_features_out))
    y_pred = pd.DataFrame(scaler.inverse_transform(np.concatenate([y_complete[-1][:,:-n_features_out], y_pred], 
    axis=1))[:,-n_features_out:],columns=names)
    y_pred['Date'] = data.index[-n_steps_out:]


    y_future = model.predict(np.reshape(dataset[-n_steps_in:], (1, n_steps_in,n_features_in, 1)))
    y_future = np.reshape(y_future[-1], (n_steps_out,n_features_out))
    y_future = pd.DataFrame(scaler.inverse_transform(np.concatenate([y_complete[-1][:,:-n_features_out],
y_future], axis=1))[:,-n_features_out:],columns=names)
    y_future['Date'] = test_date[1:]

    df_true = pd.concat([X_test[-5:],y_test],sort=False)
    df_pred = pd.concat([y_pred,y_future],sort=False)

    df_true['Date'] = pd.to_datetime(df_true['Date'])
    df_pred['Date'] = pd.to_datetime(df_pred['Date'])   
    df_true['Date'] = np.array(pd.Series(df_true['Date']).apply(lambda x: x.day).astype(str))
    df_pred['Date'] = np.array(pd.Series(df_pred['Date']).apply(lambda x: x.day).astype(str))


    # ax1.plot( df_true['Date'], df_true['High'], linewidth=2, linestyle='--',color='blue')
    # ax1.plot(df_true['Date'], df_true['Open'], linewidth=2, linestyle='--',color='yellow')
    ax3.plot(df_true['Date'], df_true['Close'], markerfacecolor='green',
         markeredgecolor='green', linewidth=2, marker='.', linestyle='--',color='yellow', label='Real Adj Close')
    # ax1.plot(df_true['Date'], df_true['Low'], linewidth=2, linestyle='--',color='red')


    # ax1.plot(df_pred['Date'], df_pred['High'], linewidth=2,marker='^', color='blue', label='High')
    # ax1.plot(df_pred['Date'], df_pred['Open'], linewidth=2, marker='*', color='yellow', label='Open')
    ax3.plot(df_pred['Date'], df_pred['Close'], markerfacecolor='white',
         markeredgecolor='yellow', linewidth=2, marker='*', color='green', label='Predited Adj Close')
    # ax1.plot(df_pred['Date'], df_pred['Low'], linewidth=2, marker='v', color='red', label='Low')

    ax3.set_title('CNN-LSTM - MAE: ' + str(round(MAE_CNN_LSTM, 1)))
    # ax3.set_xlabel('Days')
    ax3.set_ylabel('Stock Value')   

    #ax3.tick_params(labelrotation=45)
    #ax3.legend()
    #ax3.grid()

    model = load_model('models/' + stock + '-FG-ConvLSTM2D-2y.h5' )
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1, 1)
    yhat = model.predict(X[-N_samples:], verbose=0)
    MAE_ConvLSTM = mean_absolute_percentage_error(np.reshape(y[-N_samples:], (N_samples*n_steps_out*n_features_out)),
                                             np.reshape(yhat[-N_samples:], (N_samples*n_steps_out*n_features_out)))

    y_pred = np.reshape(yhat[-1], (n_steps_out,n_features_out))
    y_pred = pd.DataFrame(scaler.inverse_transform(np.concatenate([y_complete[-1][:,:-n_features_out],
     y_pred], axis=1))[:,-n_features_out:],columns=names)
    y_pred['Date'] = data.index[-n_steps_out:]


    y_future = model.predict(np.reshape(dataset[-n_steps_in:], (1, n_steps_in, n_features_in, 1, 1)))
    y_future = np.reshape(y_future[-1], (n_steps_out,n_features_out))
    y_future = pd.DataFrame(scaler.inverse_transform(np.concatenate([y_complete[-1][:,:-n_features_out],
     y_future], axis=1))[:,-n_features_out:],columns=names)
    y_future['Date'] = test_date[1:]

    df_true = pd.concat([X_test[-5:],y_test],sort=False)
    df_pred = pd.concat([y_pred,y_future],sort=False)

    df_true['Date'] = pd.to_datetime(df_true['Date'])
    df_pred['Date'] = pd.to_datetime(df_pred['Date'])   
    df_true['Date'] = np.array(pd.Series(df_true['Date']).apply(lambda x: x.day).astype(str))
    df_pred['Date'] = np.array(pd.Series(df_pred['Date']).apply(lambda x: x.day).astype(str))


   # ax1.plot( df_true['Date'], df_true['High'], linewidth=2, linestyle='--',color='blue')
    # ax1.plot(df_true['Date'], df_true['Open'], linewidth=2, linestyle='--',color='yellow')
    ax4.plot(df_true['Date'], df_true['Close'], markerfacecolor='green',
         markeredgecolor='green', linewidth=2, marker='.', linestyle='--',color='yellow', label='Real Adj Close')
    # ax1.plot(df_true['Date'], df_true['Low'], linewidth=2, linestyle='--',color='red')


    # ax1.plot(df_pred['Date'], df_pred['High'], linewidth=2,marker='^', color='blue', label='High')
    # ax1.plot(df_pred['Date'], df_pred['Open'], linewidth=2, marker='*', color='yellow', label='Open')
    ax4.plot(df_pred['Date'], df_pred['Close'], markerfacecolor='white',
         markeredgecolor='yellow', linewidth=2, marker='*', color='green', label='Predited Adj Close')
    # ax1.plot(df_pred['Date'], df_pred['Low'], linewidth=2, marker='v', color='red', label='Low')

    ax4.set_title('ConvLSTM - MAE: ' + str((round(MAE_ConvLSTM, 1))))
    # ax4.set_xlabel('Days')
    ax4.set_ylabel('Stock Value')   


    #ax4.legend()
    #ax4.grid()

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,  loc = 'lower center', ncol=3)
    plt.savefig('images/' + stock + '-2y.png')
    
    
    
    keras.backend.clear_session()
    return (stock, True)


def download_data_stocks(stock):
    data = pdr.get_data_yahoo(stock)
    data.to_csv('data/'+ stock + '.csv')
    return (stock, True)


def download_data_stocks_Intraday(stock):
    import yfinance as yf
    yf.pdr_override() # <== that's all it takes :-)

    data = pdr.get_data_yahoo(stock, start="2019-01-01", interval='60m')
    data.to_csv('data/'+ stock + '-H.csv')
    return (stock, True)


foreign_stocks = [
            '^BVSP',
            '^N100',
            'USDBRL=X',
            '^NYA',            
            '^IXIC',
            'LSE.L'
            
            ]

young_stocks = ['BIDI3.SA', 'CAML3.SA', 'CNTO3.SA', 'GNDI3.SA', 
                'HAPV3.SA', 'CRFB3.SA', 'IRBR3.SA', 'NEOE3.SA',
                'NTCO3.SA', 'SAPR11.SA']


def show_mape_models(stock):

    data = pd.read_csv('data/' + stock + '.csv')
    data.index = data['Date']

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.loc[:,'HighLoad'] = (data['High'] - data['Close']) / data['Close'] * 100.0
    data.loc[:,'Change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0

    # [['HighLoad', 'Change', 'Volume', 'Low', 'High', 'Close']]



    df_foreign = load_foreign_stocks(foreign_stocks)
    data = pd.concat([data[['HighLoad', 'Change', 'Volume', 'Low', 'High', 'Close']]], axis=1, sort=True)[:]
    data = data.dropna()
    scaler = MinMaxScaler(feature_range=[0,1])
    # define input sequence
    scaler_norm = StandardScaler()
    # define input sequence
    dataset = scaler.fit_transform(data.ffill().values)

    # choose a number of time steps
    n_steps_in, n_steps_out = 5, 3

    # convert into input/output
    X, y_complete = split_sequences(dataset, n_steps_in, n_steps_out)

    y = y_complete[:,:,-1:]

    # the dataset knows the number of features, e.g. 2
    n_features_in = X.shape[2]
    n_features_out = 1

    MAPE_ALL = []
    for i in ['CNN', 'LSTM', 'CNN-LSTM', 'ConvLSTM2D']:
    # for i in ['CNN']:
        
        
        filename = 'models/' + stock + '-FG-' + i + '-2y.h5'
        model = load_model(filename)
        

        if (i == 'CNN'):            
            yhat = model.predict(X[-10:])
        elif(i == 'LSTM'):
            yhat = model.predict(X[-10:])
        elif (i == 'CNN-LSTM'):
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
            yhat = model.predict(X[-10:], verbose=0)
        else:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1, 1)
            yhat = model.predict(X[-10:], verbose=0)

        # print(np.reshape(y[-1], (n_steps_out*n_features_out)), np.reshape(yhat[-1], ( n_steps_out*n_features_out)))

        MAPE = mean_absolute_percentage_error(np.reshape(y[-10:], (10*n_steps_out*n_features_out)),
                                    np.reshape(yhat[-10:], (10*n_steps_out*n_features_out)))
        
        print(i + ': ' + str(MAPE))
        MAPE_ALL.append(MAPE)
    if np.mean(MAPE_ALL) < 5:
        print(np.mean(MAPE_ALL))
        plot_model_predictions(stock)
    



stocks = pd.read_csv('bovespa.csv')
stocks = stocks['codigo.sa'].values

'''
stocks = ['CCRO3.SA', 'COGN3.SA', 'CSNA3.SA', 'ECOR3.SA', 'EZTC3.SA', 'LWSA3.SA', 'TOTS3.SA', 'VVAR3.SA',
'BRAP3.SA', 'PETR3.SA', 'GGBR3.SA', 'EZTC3.SA', 'MRFG3.SA', 'PSSA3.SA', 'LREN3.SA'
]
'''

# pool = Pool(8)
# var = 0
# pool.map(download_data_stocks , list(stocks))    
# pool.map(download_data_stocks, list(foreign_stocks))    

#pool.map(model_conv1_FG, list(stocks))    
#pool.map(model_LSTM_FG, list(stocks))
#pool.map(model_CNN_LSTM_2D_FG, list(stocks[:var]))
#pool.map(model_ConvLSTM2D_FG, list(stocks[:var]))
# pool.map(plot_model_predictions, list(stocks[:var]))

# print( 'Argument List:', str(sys.argv))

# for stock in foreign_stocks:
#     print(stock)
#     download_data_stocks(stock)



for stock in stocks[int(sys.argv[1]):int(sys.argv[1]) + 5]:
    print(stock)
    # download_data_stocks(stock)
    # download_data_stocks_Intraday(stock)

    #model_conv1_FG(stock)
    #model_LSTM_FG(stock)
    #model_CNN_LSTM_2D_FG(stock)
    #model_ConvLSTM2D_FG(stock)
    show_mape_models(stock)
    # plot_model_predictions(stock)

	
# 	plt.close('all')
# 	plt.clf()
# 	plt.cla()
# 	plt.ioff()
