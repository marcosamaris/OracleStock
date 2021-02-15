import os 
import pickle
import sys 
import time
import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from pandas.plotting import register_matplotlib_converters
# from pandas_datareader import data as pdr
# from pandas.plotting import register_matplotlib_converters

# import keras
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from keras import backend
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import Conv1D, Dense, LSTM, RepeatVector, TimeDistributed, ConvLSTM2D, \
                        BatchNormalization, Dropout, MaxPooling2D, UpSampling1D, Bidirectional

from keras.optimizers import Adadelta, RMSprop, Adam, Adagrad, Adamax, Nadam

from keras.callbacks import CSVLogger, EarlyStopping
import plotnine as p9


from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from multiprocessing import Pool

var_patience = 25
var_batch_size = 3

def get_stock_trend(stock, df_trends, geo):
    
    df_trends.rename(columns={"date": "Week"},  inplace=True)

    dates=[]
    i=0
    while i<len(df_trends):
        dates.append(datetime.date(int(df_trends.iloc[i]['Week'].split('-')[0]),int(df_trends.iloc[i]['Week'].split('-')[1]),int(df_trends.iloc[i]['Week'].split('-')[2])))
        i+=1

    df_trends['date'] = dates
    df_trends_req=df_trends[df_trends['date']>datetime.date(2016,1,1)]

    days=[]
    trend=[]
    i=0
    while i<len(df_trends_req):
        day=df_trends_req.iloc[i]['date']
        trend.append(df_trends_req.iloc[i][stock])
        dates=[dates for dates in (day - datetime.timedelta(n) for n in range(7))]
        dates.reverse()

        j=0
        while j<len(dates):
            days.append(dates[j])
            trend.append(df_trends_req.iloc[i][stock])
            j+=1
        i+=1

    df_trend_final = pd.DataFrame(list(zip(days,trend)), columns=['Date','trend_hit_'+geo])

    df_trend_final['Date'] = pd.to_datetime(df_trend_final['Date'])
    return(df_trend_final)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def check_stock_symbol(flag=False, companies_file='companylist.csv'):
    df = pd.read_csv(companies_file, usecols=[0])

    while flag is False:
        symbol = input('Enter a stock symbol to retrieve data from: ').upper()
        for index in range(len(df)):
            if df['Symbol'][index] == symbol:
                flag = True
    return flag, symbol



def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def get_stock_data(stock, interval):

    df = pd.read_csv('data/' + stock  + '-' + interval + '.csv')
    
    df['HighLoad'] = (df['High'] - df['Close']) / df['Close'] * 100.0
    df['Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    return df
    
def stock_forecasting(dataframe, n_steps_out):
    df = dataframe

    forecast_col = 'Close'
    df['Label'] = dataframe[[forecast_col]].shift(-n_steps_out)
    X = np.array(dataframe.drop(['Label'], axis=1))
    
    

    scaler = MinMaxScaler(feature_range=[0,1])
    X = scaler.fit_transform(X)

    X_forecast = X[-n_steps_out:]
    X = X[:-n_steps_out]

    y = scaler.fit_transform(np.log1p(np.array(df['Label'])))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    predictions = clf.predict(X_forecast)

    
    return predictions, accuracy

def stock_forecastingLog(dataframe, n_steps_out):
    df = dataframe

    forecast_col = 'Close'
    df['Label'] = dataframe[[forecast_col]].shift(-n_steps_out)
    
    df['HighLoad'] = np.log(np.array(dataframe[['HighLoad']]) + 0.00000001)
    
    df['Change'] = np.log(np.array(dataframe[['Change']]) + 0.00000001)
    
    df['Volume'] = np.log(np.array(dataframe[['Volume']]) + 0.00000001)
    
    
    X = np.array(dataframe.drop(['Label'], axis=1))
    
    

    scaler = MinMaxScaler(feature_range=[0,1])
    X = scaler.fit_transform(X)

    X_forecast = X[-n_steps_out:]
    X = X[:-n_steps_out]
    df = dataframe.dropna(inplace=False)
    y = np.log(np.array(df['Label']))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    predictions = clf.predict(X_forecast)

    return predictions, accuracy


def forecast_plot(predictions, n_steps_out):


    plt.style.use(['ggplot'])

    (fig, ax1) = plt.subplots(1, 1)

    ax1.plot(predictions, color='black')
#     ax1.plot(predictions[-n_steps_out,  color='green')
    ax1.legend(loc=4)
    ax1.set_ylabel('Stock Value')  
    ax1.set_xlabel('Date')  
    ax1.tick_params(labelrotation=45)
    
    
    plt.savefig('../OracleStock/images/' + stock + '-twit' + '.png')




def predictions_max_index(predictions, n_steps_out):
    gain_pred, index_maximo_pred = 0, 0
    
    if predictions[-n_steps_out-1] < np.max(predictions[-n_steps_out:]):
        
        actual_price = predictions[-n_steps_out-1]
        maximo_pred = np.max(predictions[-n_steps_out:])  
        index_maximo_pred = np.argmax(predictions[-n_steps_out:])
        gain_pred = ((maximo_pred - actual_price)/actual_price)*100
            
    return (gain_pred, index_maximo_pred)



def recommending(predictions, n_steps_out, global_polarity, symbol, accuracy, ML_tech):
    gain_pred, index_maximo_pred = 0, 0
    
    if predictions[-n_steps_out-1] < np.max(predictions[-n_steps_out:]):
        if global_polarity > 0:
            actual_price = predictions[-n_steps_out-1]
            maximo_pred = np.max(predictions[-n_steps_out:])  
            index_maximo_pred = np.argmax(predictions[-n_steps_out:])
            gain_pred = ((maximo_pred - actual_price)/actual_price)*100


            print(symbol + " , " + ML_tech + ' , ' + str(round(accuracy,2)) + "," + str(round(global_polarity,2)) + "," + str(round(gain_pred,1)) + "," + str(index_maximo_pred))

            
    return (gain_pred, index_maximo_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def data_grouped_foreign_stock(foreign_stocks, interval):

    dfm = pd.DataFrame()
    for stock_foeign in foreign_stocks:    
        dataframe = pd.read_csv('./data/' + stock_foeign  + '-' + interval + '.csv')
        dataframe['Stock'] = stock_foeign
        
        df2 = dataframe[['Date', 'Stock', 'Adj Close']]
        dfm = pd.concat([dfm, df2], axis=0)
        
    dfm.index = pd.to_datetime(dfm['Date'])
    datagrouped = dfm[['Stock', 'Adj Close']].groupby([pd.Grouper(freq='1d'), 'Stock'])['Adj Close'].mean().unstack()
    return datagrouped

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


def predict_conv1D(stock, X, interval, n_steps_in):
    
    filename = 'models/' + stock + '-Conv1D-' + interval +'.h5'
    
    
    if os.path.isfile(filename):       
        model = load_model(filename)        
        
        predictions = model.predict(X)
        return predictions
        
    else:
        print('Machine Learning  file does not exist')

def predict_LSTM(stock, X, interval, n_steps_in):
    
    filename = 'models/' + stock + '-LSTM-' + interval +'.h5'
    
    if os.path.isfile(filename):       
        model = load_model(filename)        
        
        predictions = model.predict(X)
        return predictions
        
    else:
        print('Machine Learning  file does not exist')      
        

def predict_BidirectionalLSTM(stock, X, interval, n_steps_in):
    
    filename = 'models/' + stock + '-BidirectionalLSTM-' + interval +'.h5'
    
    if os.path.isfile(filename):       
        model = load_model(filename)        
        
        predictions = model.predict(X)
        return predictions
        
    else:
        print('Machine Learning  file does not exist')   
    

    
def plot_bar_predictions(data, filenamePlot, x, y, facet,plot_size):
    # dados = dados.loc[dados['Gain'] < 25]

    var_plot_bar_all_predictions = p9.ggplot(data, p9.aes(x=x, y=y)) +\
      p9.geom_bar(stat='identity') +\
      p9.geom_text(p9.aes(label=y),size=7, va='bottom') +\
      p9.facet_wrap(facet) +\
      p9.scales.scale_y_log10() +\
      p9.theme(axis_text_x = p9.element_text(angle=90, size =7.5 )) +\
      p9.theme(subplots_adjust={'wspace': 0.25})

    p9.ggsave(var_plot_bar_all_predictions, 'images/plot_Bar_' + filenamePlot + '.png',height=plot_size, width=plot_size, units = 'in', dpi=300)
    return var_plot_bar_all_predictions



def predict_convLSTM1D(stock, X, interval, n_steps_in):
    
    filename = 'models/' + stock + '-ConvLSTM1D-' + interval +'.h5'
    
    if os.path.isfile(filename):       
        model = load_model(filename)        
        
        predictions = model.predict(X)
        return predictions
        
    else:
        print('Machine Learning  file does not exist')   
        
def predict_convLSTM2D(stock, X, interval, n_steps_in):
    
    filename = 'models/' + stock + '-ConvLSTM2D-' + interval +'.h5'
    
    if os.path.isfile(filename):       
        model = load_model(filename)        
        
        predictions = model.predict(X)
        return predictions
        
    else:
        print('Machine Learning  file does not exist')   
        


def model_conv1D(stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose):
    
    filename = 'models/' + stock + '-Conv1D-' + interval +'.h5'
    
    if os.path.isfile(filename) and (update == False):       
        model = load_model(filename)
    else:
        
        n_output = y.shape[1] * y.shape[2]
        
        
        
        # the dataset knows the number of features, e.g. 2
        n_features = X.shape[2]

        # define model
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(Dropout(0.25))


        model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
        model.add(Dropout(0.25))

        

        model.add(Flatten())
        model.add(Dense(10, activation='relu'))
        
        model.add(Dense(n_output))
        opt = 'adam'

        model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['mean_absolute_error'])

        early_stop = EarlyStopping(monitor='val_loss', patience=var_patience)    
        logdir = 'logs/' + stock + '-Conv1D-' + interval + '-' + str(n_steps_in) +'-'+ str(n_steps_out) +'.log'

        early_stop = EarlyStopping(monitor='val_loss', patience=var_patience)
        csv_logger = CSVLogger(logdir,separator=",",append=True)    

        # fit model
        model.fit(X, y, epochs=epochs, verbose=verbose,      batch_size=var_batch_size, validation_split=0.1,   callbacks=[early_stop, csv_logger])

        if save:
            model.save(filename)
        
    return (stock, model)

def model_LSTM(stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose):
    
    filename = 'models/' + stock + '-LSTM-' + interval +'.h5'
    
    if os.path.isfile(filename) and (update == False):       
        model = load_model(filename)
    else:
        
        n_output = y.shape[1] * y.shape[2]
        n_features_in = X.shape[2]
        n_features_out = y.shape[2]

        # define model
        model = Sequential()
        model.add(LSTM(32, activation='relu', input_shape=(n_steps_in, n_features_in)))
        model.add(RepeatVector(n_steps_out))
        model.add(LSTM(32, activation='relu'))
        model.add(RepeatVector(n_steps_out))

        model.add(LSTM(32, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(10, activation='relu')))
        model.add(TimeDistributed(Dense(n_features_out)))
        opt = 'adam'
        model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['mean_absolute_error'])

        early_stop = EarlyStopping(monitor='val_loss', patience=var_patience)    
        logdir = 'logs/' + stock + '-LSTM-' + interval + '-' + str(n_steps_in) +'-'+ str(n_steps_out) +'.log'

        early_stop = EarlyStopping(monitor='val_loss', patience=var_patience)
        csv_logger = CSVLogger(logdir,separator=",",append=True)    

        # fit model
        model.fit(X, y, epochs=epochs, verbose=verbose,      batch_size=var_batch_size, validation_split=0.1,   callbacks=[early_stop, csv_logger])

        if save:
            model.save(filename)
        
    return (stock, model)

def model_BidirectionalLSTM(stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose):
    
    filename = 'models/' + stock + '-BidirectionalLSTM-' + interval +'.h5'
    
    if os.path.isfile(filename) and (update == False):       
        model = load_model(filename)
    else:
        
        n_output = y.shape[1] * y.shape[2]
        n_features_in = X.shape[2]
        n_features_out = y.shape[2]

        # define model
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps_in, n_features_in)))
        
        model.add(Dense(n_steps_out))
        opt = 'adam'
        
        model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['mean_absolute_error'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=var_patience)    
        logdir = 'logs/' + stock + '-BidirectionalLSTM-' + interval + '-' + str(n_steps_in) +'-'+ str(n_steps_out) +'.log'

        early_stop = EarlyStopping(monitor='val_loss', patience=var_patience)
        csv_logger = CSVLogger(logdir,separator=",",append=True)    

        # fit model
        model.fit(X, y, epochs=epochs, verbose=verbose,      batch_size=var_batch_size, validation_split=0.1,   callbacks=[early_stop, csv_logger])

        if save:
            model.save(filename)
        
    return (stock, model)



def model_convLSTM1D(stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose):

    filename = 'models/' + stock + '-ConvLSTM1D-' + interval +'.h5'
    
    if os.path.isfile(filename) and (update == False):       
        model = load_model(filename)
    else:

        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        # flatten output
        n_output = y.shape[1] * y.shape[2]

        n_features_in = X.shape[2]

        # define model
        model = Sequential()

        model.add(TimeDistributed(Conv1D(filters=32, kernel_size=1, activation='relu'), 
                                  input_shape=(n_steps_in, n_features_in, 1)))

        model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu')))                            

        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(10, activation='relu'))

        model.add(Dense(n_output))
        opt = 'adam'

        model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['mean_absolute_error'])
        early_stop = EarlyStopping(monitor='val_loss', patience=var_patience)    
        logdir = 'logs/' + stock + '-ConvLSTM1D-' + interval + '-' + str(n_steps_in) +'-'+ str(n_steps_out) +'.log'

        early_stop = EarlyStopping(monitor='val_loss', patience=var_patience)
        csv_logger = CSVLogger(logdir,separator=",",append=True)    

        # fit model
        model.fit(X, y, epochs=epochs, verbose=verbose,      batch_size=var_batch_size, validation_split=0.1,   callbacks=[early_stop, csv_logger])

        if save:
            model.save(filename)
        
    return (stock, model)



def model_ConvLSTM2D(stock, X, y, interval, n_steps_in, n_steps_out, epochs, save, update, verbose):

    filename = 'models/' + stock + '-ConvLSTM2D-' + interval +'.h5'
    
    if os.path.isfile(filename) and (update == False):       
        model = load_model(filename)
    else:
        

        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1, 1)
        

        # flatten output
        n_output = y.shape[1] * y.shape[2]

        n_features_in = X.shape[2]
        n_features_out = y.shape[1]

        # define model
        model = Sequential()
        model.add(ConvLSTM2D(filters=16, kernel_size=(1,1), activation='relu', 
                             input_shape=(n_steps_in, n_features_in, 1, 1),
                             padding = 'same', return_sequences = True))

        model.add(Dropout(0.2, name = 'dropout_1')) 
        model.add(BatchNormalization(name = 'batch_norm_1'))
        model.add(ConvLSTM2D(filters=64, kernel_size=(3,1), activation='relu'))


        model.add(Flatten())

        #model.add(TimeDistributed(Dense(units=1, name = 'dense_1', activation = 'relu')))
        
        model.add(Dense(10, activation='relu'))

        model.add(Dense(n_features_out))
        opt = 'adam'

        model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['mean_absolute_error'])
        early_stop = EarlyStopping(monitor='val_loss', patience=var_patience)    
        logdir = 'logs/' + stock + '-ConvLSTM2D-' + interval + '-' + str(n_steps_in) +'-'+ str(n_steps_out) +'.log'

        early_stop = EarlyStopping(monitor='val_loss', patience=var_patience)
        csv_logger = CSVLogger(logdir,separator=",",append=True)    

        # fit model
        model.fit(X, y, epochs=epochs, verbose=verbose,      batch_size=var_batch_size, validation_split=0.1,   callbacks=[early_stop, csv_logger])

        if save:
            model.save(filename)
        
    return (stock, model)


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
    

