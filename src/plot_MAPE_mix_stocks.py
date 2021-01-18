import sys

import numpy as np
import pandas as pd
import functions as DLmodels          

    
dados = pd.read_csv('./logs/dados-MAPE-mix.csv')

filenamePlot = 'self_MAPE__mix_Less10'
var_plot_bar_all_predictions = DLmodels.plot_bar_predictions(dados.loc[dados['MAPE'] < 5], filenamePlot,'ML', 'MAPE', 'Stock', 10)


filenamePlot = 'self_MAPE__mix_High10'
# var_plot_bar_all_predictions = DLmodels.plot_bar_predictions(dados.loc[dados['MAPE'] > 10], filenamePlot,'ML', 'MAPE', 'Stock', 20)
