import time
import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from pandas_datareader import data as pdr


register_matplotlib_converters()

# import keras
from tensorflow import keras

from keras import backend
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import Conv1D, Dense, LSTM, RepeatVector, TimeDistributed, ConvLSTM2D, BatchNormalization, Dropout
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


def load_foreign_stocks(foreign_stocks, feature = 'Adj Close'):

    df_foreign = pd.DataFrame()

    for stock in foreign_stocks:
        temp_df = pd.read_csv('./data/' + stock + '.csv')
        temp_df.index = temp_df['Date']
    
        df_foreign = pd.concat([df_foreign, temp_df[[feature]]], axis=1, sort=True)
        df_foreign = df_foreign.dropna()                                             


    return df_foreign
    

def model_conv1_FG(stock):
    
    data = pd.read_csv('data/' + stock + '.csv')
    data.index = data['Date']
    
    df_foreign = load_foreign_stocks(foreign_stocks) 

    data = pd.concat([df_foreign.iloc[:, 0:2],data[['Volume', 'Low', 'Adj Close', 'High']]], axis=1, sort=True)[:-5]
    data = data.dropna()
    
    scaler = MinMaxScaler(feature_range=[0,1])
    # define input sequence
    dataset = scaler.fit_transform(data.ffill().values)

    # choose a number of time steps
    n_steps_in, n_steps_out = 10, 5
    
    # convert into input/output
    X, y = split_sequences(dataset, n_steps_in, n_steps_out)
             
    # flatten output
    y = y[:,:,-3:]
    
    n_output = y.shape[1] * y.shape[2]
    y = y.reshape((y.shape[0], n_output))
    
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(n_steps_in, n_features)))

    #model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(n_output))

    model.compile(optimizer='adam', loss='mse')

    # fit model
    early_stop = EarlyStopping(monitor='val_loss', patience=200)
    model.fit(X, y, epochs=2000, verbose=0, validation_split=0.2,  callbacks=[early_stop])
    
    model.save('models/' + stock + '-FG-CNN.h5')
    
    return (stock, True)

def model_LSTM_FG(stock):
    
    data = pd.read_csv('data/' + stock + '.csv')
    data.index = data['Date']

    df_foreign = load_foreign_stocks(foreign_stocks) 
    
    data = pd.concat([df_foreign.iloc[:, 0:2],data[['Volume', 'Low', 'Adj Close', 'High']]], axis=1, sort=True)[:-5]
    data = data.dropna()
    
    scaler = MinMaxScaler(feature_range=[0,1])
    # define input sequence
    dataset = scaler.fit_transform(data.ffill().values)

    # choose a number of time steps
    n_steps_in, n_steps_out = 10, 5
    # convert into input/output
    X, y = split_sequences(dataset, n_steps_in, n_steps_out)
                           
    # flatten output
    y = y[:,:,-3:]
    
    n_output = y.shape[1] * y.shape[2]
    n_features_in = X.shape[2]
    n_features_out = y.shape[2]
    
    # define model
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(n_steps_in, n_features_in)))
    model.add(RepeatVector(n_steps_out))
    
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(n_features_out)))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    early_stop = EarlyStopping(monitor='val_loss', patience=200)
    model.fit(X, y, epochs=2000, verbose=0, validation_split=0.2,  callbacks=[early_stop])
    
    model.save('models/' + stock + '-FG-LSTM.h5')    
    
    return (stock, True)


def model_CNN_LSTM_2D_FG(stock):

    data = pd.read_csv('data/' + stock + '.csv')
    data.index = data['Date']

    df_foreign = load_foreign_stocks(foreign_stocks) 

    data = pd.concat([df_foreign.iloc[:, 0:2],data[['Volume', 'Low', 'Adj Close', 'High']]], axis=1, sort=True)[:-5]

    data = data.dropna()
    
    scaler = MinMaxScaler(feature_range=[0,1])
    # define input sequence
    dataset = scaler.fit_transform(data.ffill().values)

    # choose a number of time steps
    n_steps_in, n_steps_out = 10, 5
    # convert into input/output
    X, y_complete = split_sequences(dataset, n_steps_in, n_steps_out)

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = y_complete[:,:,-3:]

    # flatten output
    n_output = y.shape[1] * y.shape[2]

    y = y.reshape((y.shape[0], n_output))

    n_features_in = X.shape[2]
    
    # define model
    model = Sequential()

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), 
                              input_shape=(n_steps_in, n_features_in, 1)))

    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(1000, activation='relu'))
    model.add(Dense(n_output))

    model.compile(optimizer='adam', loss='mse')
#     print(model.summary())
    
    # fit model
    early_stop = EarlyStopping(monitor='val_loss', patience=200)

    model.fit(X, y, epochs=2000, verbose=0, validation_split=0.2,  callbacks=[early_stop])
    
    model.save('models/' + stock + '-FG-CNN-LSTM.h5')
        
    return (stock, True)


def model_ConvLSTM2D_FG(stock):

    data = pd.read_csv('data/' + stock + '.csv')
    data.index = data['Date']

    df_foreign = load_foreign_stocks(foreign_stocks) 

    data = pd.concat([df_foreign.iloc[:, 0:2],data[['Volume', 'Low', 'Adj Close', 'High']]], axis=1, sort=True)[:-5]

    data = data.dropna()

    scaler = MinMaxScaler(feature_range=[0,1])
    
    # define input sequence
    dataset = scaler.fit_transform(data.ffill().values)
	
    # choose a number of time steps
    n_steps_in, n_steps_out = 10, 5
    # convert into input/output
    X, y_complete = split_sequences(dataset, n_steps_in, n_steps_out)

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1, 1)
    y = y_complete[:,:,-3:]

    # flatten output
    n_output = y.shape[1] * y.shape[2]

    y = y.reshape((y.shape[0], n_output))

    n_features_in = X.shape[2]
    n_features_out = y.shape[1]

    # define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,1), activation='relu', 
                         input_shape=(n_steps_in, n_features_in, 1, 1)))


                         

    model.add(Flatten())

    #model.add(TimeDistributed(Dense(units=1, name = 'dense_1', activation = 'relu')))
    
    model.add(Dense(n_features_out))

    model.compile(optimizer='adam', loss='mse')

    # fit model
    early_stop = EarlyStopping(monitor='val_loss', patience=200)
    
    model.fit(X, y, epochs=2000, verbose=0, validation_split=0.2,  callbacks=[early_stop])
    
    model.save('models/' + stock + '-FG-ConvLSTM2D.h5')
    
    return (stock, True)


def plot_model_predictions(stock):
	
    data = pd.read_csv('data/' + stock + '.csv')
    data.index = data['Date']

    df_foreign = load_foreign_stocks(foreign_stocks) 

    data = pd.concat([df_foreign.iloc[:, 0:2],data[['Volume', 'Low', 'Adj Close', 'High']]], axis=1, sort=True)[:-1]
    data = data.dropna()
    
    scaler = MinMaxScaler(feature_range=[0,1])
    # define input sequence
    dataset = scaler.fit_transform(data.ffill().values)

    # choose a number of time steps
    n_steps_in, n_steps_out = 10, 5
    
    # convert into input/output
    X, y_complete = split_sequences(dataset, n_steps_in, n_steps_out)

    y = y_complete[:,:,-3:]

    # the dataset knows the number of features, e.g. 2
    n_features_in = X.shape[2]
    n_features_out = 3
       

    names = ['Low', 'Adj Close', 'High']
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
    
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(9,7), sharex=True,sharey=False)
    
    model = load_model('models/' + stock + '-FG-CNN.h5' )
    yhat = model.predict(X, verbose=0)
    MAE_CNN = mean_absolute_percentage_error(np.reshape(y[-1], (n_steps_out*n_features_out)),
                                np.reshape(yhat[-1], ( n_steps_out*n_features_out)))
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

    fig.suptitle(stock +  ' daily prediction. Solid lines = Predicted')

    ax1.plot( df_true['Date'], df_true['High'], linewidth=2, linestyle='--',color='blue')
    # ax1.plot(df_true['Date'], df_true['Open'], linewidth=2, linestyle='--',color='yellow')
    ax1.plot(df_true['Date'], df_true['Adj Close'], linewidth=2,linestyle='--',color='black')
    ax1.plot(df_true['Date'], df_true['Low'], linewidth=2, linestyle='--',color='red')
    

    ax1.plot(df_pred['Date'], df_pred['High'], linewidth=2,marker='^', color='blue', label='High')
    # ax1.plot(df_pred['Date'], df_pred['Open'], linewidth=2, marker='*', color='yellow', label='Open')
    ax1.plot(df_pred['Date'], df_pred['Adj Close'], linewidth=2, marker='.', color='black', label='Adj Close')
    ax1.plot(df_pred['Date'], df_pred['Low'], linewidth=2, marker='v', color='red', label='Low')
    
    ax1.set_title('CNN - MAE: ' + "{:.3}".format(round(MAE_CNN, 3)))  
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Stock Value')   

    ax1.tick_params(labelrotation=45)
    ax1.legend()
    ax1.grid()
    
    model = load_model('models/' + stock + '-FG-LSTM.h5' )
    yhat = model.predict(X, verbose=0)
    MAE_LSTM = mean_absolute_percentage_error(np.reshape(y[-1], (n_steps_out*n_features_out)),
                                np.reshape(yhat[-1], ( n_steps_out*n_features_out)))
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

    ax2.plot( df_true['Date'], df_true['High'], linewidth=2, linestyle='--',color='blue')
    # ax2.plot(df_true['Date'], df_true['Open'], linewidth=2, linestyle='--',color='yellow')
    ax2.plot(df_true['Date'], df_true['Adj Close'], linewidth=2,linestyle='--',color='black')
    ax2.plot(df_true['Date'], df_true['Low'], linewidth=2, linestyle='--',color='red')
    

    ax2.plot(df_pred['Date'], df_pred['High'], linewidth=2,marker='^', color='blue', label='High')
    # ax2.plot(df_pred['Date'], df_pred['Open'], linewidth=2, marker='*', color='yellow', label='Open')
    ax2.plot(df_pred['Date'], df_pred['Adj Close'], linewidth=2, marker='.', color='black', label='Adj Close')
    ax2.plot(df_pred['Date'], df_pred['Low'], linewidth=2, marker='v', color='red', label='Low')
    
    ax2.set_title('LSTM - MAE: ' + "{:.3}".format(round(MAE_LSTM, 3)))
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Stock Value')   

    ax2.tick_params(labelrotation=45)
    ax2.legend()
    ax2.grid()
    
    model = load_model('models/' + stock + '-FG-CNN-LSTM.h5' )
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    yhat = model.predict(X, verbose=0)
    MAE_CNN_LSTM = mean_absolute_percentage_error(np.reshape(y[-1], (n_steps_out*n_features_out)),
                                np.reshape(yhat[-1], ( n_steps_out*n_features_out)))
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

    ax3.plot( df_true['Date'], df_true['High'], linewidth=2, linestyle='--',color='blue')
    # ax3.plot(df_true['Date'], df_true['Open'], linewidth=2, linestyle='--',color='yellow')
    ax3.plot(df_true['Date'], df_true['Adj Close'], linewidth=2,linestyle='--',color='black')
    ax3.plot(df_true['Date'], df_true['Low'], linewidth=2, linestyle='--',color='red')
    
    ax3.plot(df_pred['Date'], df_pred['High'], linewidth=2,marker='^', color='blue', label='High')
    # ax3.plot(df_pred['Date'], df_pred['Open'], linewidth=2, marker='*', color='yellow', label='Open')
    ax3.plot(df_pred['Date'], df_pred['Adj Close'], linewidth=2, marker='.', color='black', label='Adj Close')
    ax3.plot(df_pred['Date'], df_pred['Low'], linewidth=2, marker='v', color='red', label='Low')

    ax3.set_title('CNN-LSTM - MAE: ' + "{:.3}".format(round(MAE_CNN_LSTM, 3)))
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Stock Value')   

    ax3.tick_params(labelrotation=45)
    ax3.legend()
    ax3.grid()
    
    
    model = load_model('models/' + stock + '-FG-ConvLSTM2D.h5' )
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1, 1)
    yhat = model.predict(X, verbose=0)
    MAE_ConvLSTM = mean_absolute_percentage_error(np.reshape(y[-1], (n_steps_out*n_features_out)),
                                np.reshape(yhat[-1], ( n_steps_out*n_features_out)))
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

    ax4.plot( df_true['Date'], df_true['High'], linewidth=2, linestyle='--',color='blue')
    # ax4.plot(df_true['Date'], df_true['Open'], linewidth=2, linestyle='--',color='yellow')
    ax4.plot(df_true['Date'], df_true['Adj Close'], linewidth=2,linestyle='--',color='black')
    ax4.plot(df_true['Date'], df_true['Low'], linewidth=2, linestyle='--',color='red')
    

    ax4.plot(df_pred['Date'], df_pred['High'], linewidth=2,marker='^', color='blue', label='High')
    # ax4.plot(df_pred['Date'], df_pred['Open'], linewidth=2, marker='*', color='yellow', label='Open')
    ax4.plot(df_pred['Date'], df_pred['Adj Close'], linewidth=2, marker='.', color='black', label='Adj Close')
    ax4.plot(df_pred['Date'], df_pred['Low'], linewidth=2, marker='v', color='red', label='Low')
    
    ax4.set_title('ConvLSTM - MAE: ' + "{:.3}".format(round(MAE_ConvLSTM, 3)))
    ax4.set_xlabel('Days')
    ax4.set_ylabel('Stock Value')   

    ax4.tick_params(labelrotation=45)
    ax4.legend()
    ax4.grid()

    plt.savefig('images/' + stock + '.png', dpi=150)
    return (stock, True)


def download_data_stocks(stock):
    data = pdr.get_data_yahoo(stock, start='2017/12/16')
    data.to_csv('data/'+ stock + '.csv')
    return (stock, True)

    
foreign_stocks = [
            '^BVSP',
            '^NYA',
            '^IXIC',
            'LSE.L',
            '^N100',
            '2104.TW',
            'USDBRL=X'
            ]
        
stocks = pd.read_csv('bovespa.csv')
stocks = stocks['codigo.sa'].values

#stocks = ['CCRO3.SA', 'COGN3.SA', 'CSNA3.SA', 'ECOR3.SA', 'EZTC3.SA', 'LINX3.SA', 'TOTS3.SA', 'VVAR3.SA']
# if __name__ == '__main__':
	
pool = Pool(5)
var = len(stocks)
pool.map(download_data_stocks , list(stocks))    
pool.map(download_data_stocks, list(foreign_stocks))    

#pool.map(model_conv1_FG, list(stocks[:var]))    
#pool.map(model_LSTM_FG, list(stocks[:var]))
#pool.map(model_CNN_LSTM_2D_FG, list(stocks[:var]))
#pool.map(model_ConvLSTM2D_FG, list(stocks[:var]))
# pool.map(plot_model_predictions, list(stocks[:var]))

import tensorflow as tf 

if tf.test.gpu_device_name(): 
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


#for stock in stocks[:var]:

# # # model_conv1_FG(stock)
# # # model_LSTM_FG(stock)
#    model_CNN_LSTM_2D_FG(stock)
#    model_ConvLSTM2D_FG(stock)

#    plot_model_predictions(stock)


