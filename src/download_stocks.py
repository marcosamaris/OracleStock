#eu mandei e-mail pra Leliane e copiei vc! Vc poderia dar uma cutucada nela pra saber o que ela poderia!

import datetime as dt
import yfinance as yf
import pandas as pd

def get_download_data(symbol, from_date, interval):
    import yfinance as yf
    yf.pdr_override() # <== that's all it takes :-)

    data = yf.download(symbol, interval=interval, rounding=True, start=from_date)

    return data

years = 5
actual_date = dt.date.today()
past_date = actual_date - dt.timedelta(days=365*years)

actual_date = actual_date.strftime("%Y-%m-%d")
past_date = past_date.strftime("%Y-%m-%d")


# interval='1d'
interval='1wk'
foreign_stocks = [
            '^BVSP',
            '^N100',
            'USDBRL=X',
            '^NYA',            
            '^IXIC',
            'LSE.L'           
            ]
stocks = pd.read_csv('bovespa.csv')
stocks = stocks['codigo'].values

for interval in ['1d', '1wk']:

    for stock in foreign_stocks:    
      dataframe = get_download_data(stock, past_date, interval)
      dataframe.to_csv('./data/' + stock + '-' + interval + '.csv')


    for stock in stocks:    
       dataframe = get_download_data(stock + '.SA', past_date, interval)
       dataframe.to_csv('./data/' + stock + '-' + interval + '.csv')