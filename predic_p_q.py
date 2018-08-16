import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from statsmodels.tsa.arima_model import ARIMA
from sys import maxsize

df = open('flow1200.csv',)
data = pd.read_csv(df,encoding='utf-8', index_col='time')
data.index = pd.to_datetime(data.index)
data_log = np.log(data)

def proper_model(data_ts, maxLag):
    init_bic = maxsize
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARIMA(data_ts, order=(p,0,q))
            try:
                results_ARIMA = model.fit(disp=-1)
            except:
                continue
            bic = results_ARIMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARIMA
                init_bic = bic
    return init_bic, init_p, init_q, init_properModel

if __name__ == '__main__':

    print(proper_model(data_log,3))