import sys

import numpy as np
import pandas as pd
import functions as DLmodels          

    
dados = pd.read_csv('./logs/dados-MAPE.csv')

threshold = 5

filenamePlot = 'self_MAPE_Less' + str(threshold)
var_plot_bar_all_predictions = DLmodels.plot_bar_predictions(dados.loc[dados['MAPE'] < threshold], filenamePlot,'ML', 'MAPE', 'Name', 10)


filenamePlot = 'self_MAPE_High'+ str(threshold)
var_plot_bar_all_predictions = DLmodels.plot_bar_predictions(dados.loc[dados['MAPE'] > threshold], filenamePlot,'ML', 'MAPE', 'Name', 20)


