#eu mandei e-mail pra Leliane e copiei vc! Vc poderia dar uma cutucada nela pra saber o que ela poderia!

import datetime as dt
import yfinance as yf
import pandas as pd
import constants_stocks as cs
import yfinance as yf

def get_download_data(symbol, from_date, interval):    
    yf.pdr_override() # <== that's all it takes :-)
    data = yf.download(symbol, interval=interval, rounding=True, start=from_date)
    return data

years = 3
actual_date = dt.date.today()
past_date = actual_date - dt.timedelta(days=365*years)

actual_date = actual_date.strftime("%Y-%m-%d")
past_date = past_date.strftime("%Y-%m-%d")


for interval in ['1d', '1wk']:

    for stock in cs.foreign_stocks:    
      dataframe = get_download_data(stock, past_date, interval)
      dataframe.to_csv('./data/' + stock + '-' + interval + '.csv')

    for stock in cs.stocks_codigo:    
       dataframe = get_download_data(stock + '.SA', past_date, interval)
       dataframe.to_csv('./data/' + stock + '-' + interval + '.csv')
