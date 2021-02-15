import sys

import numpy as np
import pandas as pd
import constants_stocks as cs
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pickle
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

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


n_steps_in, n_steps_out = 7, 1

interval='1wk'
samples_test = 5
all_MAPE = []
ML_tech = 'LevenbergMarquardt'
for stock in cs.stocks_codigo[int(sys.argv[1]):int(sys.argv[1]) + len(cs.stocks_codigo)]:
    print(stock)
    (flag, symbol) = (True, stock)
    if flag:

        dataframe = get_stock_data(symbol, interval)

        dataset = dataframe[['Open', 'High', 'Low', 'Close', 'Volume', 'HighLoad', 'Change', 'Adj Close']].ffill().values

        scaler = StandardScaler()
        dataset = scaler.fit_transform(dataset)

        scaler_filename = 'scalers/' + stock + '-' + interval + '.save'
        scaler = pickle.load(open(scaler_filename, 'rb'))
        dataset = scaler.fit_transform(dataset)

        X, y = split_sequences(dataset, n_steps_in, n_steps_out)
        n_features = X.shape[2]
        y = y[:,:,-1:]

        x_test = X[-samples_test:].reshape(len(X[-samples_test:]), n_steps_in*n_features)

        y_test = y[-samples_test:].reshape(len(X[-samples_test:]), n_steps_out)

        filename = 'models/' + stock + '-LevenbergMarquardt-' + interval +'.pickle'
        model = pickle.load(open(filename, 'rb'))
        
        predictions = model.predict(x_test)
        
        MAPE = mean_absolute_percentage_error(np.reshape(y_test, (samples_test*n_steps_out,)), np.reshape(predictions, (samples_test*n_steps_out,)))

        all_MAPE.append([stock, ML_tech, np.around(MAPE,1), n_steps_in, n_steps_out, interval, samples_test])


        dados = pd.DataFrame(all_MAPE)

        dados.columns = ['Stock', 'ML', 'MAPE', 'n_steps_in', 'n_steps_out', 'interval', 'samples_test']

        dados.to_csv('./logs/data_MAPE_self_LevenbergMarquardt.csv')

