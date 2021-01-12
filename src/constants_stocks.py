import pandas as pd

stocks = pd.read_csv('bovespa.csv')
stocks_codigo = stocks['codigo'].values

foreign_stocks = [
            '^BVSP',
            '^N100',
            'USDBRL=X',
            '^NYA',            
            '^IXIC',
            'LSE.L'           
            ]



