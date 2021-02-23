import numpy as np

import pandas as pd

import constants_stocks as cs
from pytrends.request import TrendReq


dataset = []
for stock in cs.stocks_codigo:
    print(stock)
    pytrends = TrendReq(hl='pt-BR', tz=360)
    pytrends.build_payload([stock], cat=0, timeframe='all', geo='BR', gprop='')
    data = pytrends.interest_over_time()

    if not data.empty:
        data = data.drop(labels=['isPartial'],axis='columns')
        dataset.append(data)

result = pd.concat(dataset, axis=1)
result.rename(columns={"date": "Date"},  inplace=True)
result.to_csv('logs/trends_pt-BR.csv')




dataset = []
for stock in cs.stocks_codigo:
    print(stock)

    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([stock], cat=0, timeframe='all', geo='US', gprop='')
    data = pytrends.interest_over_time()

    if not data.empty:
        data = data.drop(labels=['isPartial'],axis='columns')
        dataset.append(data)

result = pd.concat(dataset, axis=1)
result.rename(columns={"date": "Date"},  inplace=True)
result.to_csv('logs/trends_en-US.csv')




trends_br = pd.read_csv('logs/trends_pt-BR.csv')
trends_us = pd.read_csv('logs/trends_en-US.csv')


for stock in cs.stocks_codigo:
    if not (stock in trends_us.columns):
        trends_us[stock] = 0
    

    if not (stock in trends_br.columns):
        trends_br[stock] = 0

trends_br.to_csv('logs/trends_pt-BR.csv')
trends_us.to_csv('logs/trends_en-US.csv')


