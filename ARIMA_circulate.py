from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA



def diff_ts(ts, d):
    global shift_ts_list
    #  动态预测第二日的值时所需要的差分序列
    global last_data_shift_list
    shift_ts_list = []
    last_data_shift_list = []
    tmp_ts = ts
    tmp_ts_diff = tmp_ts - tmp_ts.shift()
    tmp_ts_diff.dropna(inplace=True)
    return tmp_ts_diff
# 还原操作
def predict_diff_recover(predict_value, d):
    if isinstance(predict_value, float):
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    elif isinstance(predict_value, np.ndarray):
        tmp_data = predict_value[0]
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    else:
        tmp_data = predict_value
        for i in range(len(d)):
            try:
                tmp_data = tmp_data.add(shift_ts_list[-i-1])
            except:
                raise ValueError('What you input is not pd.Series type!')
        tmp_data.dropna(inplace=True)
    return tmp_data

def _add_new_data(ts, dat, type='day'):
    if type == 'day':
        new_index = ts.index[-1] + relativedelta(days=1)
    elif type == 'month':
        new_index = ts.index[-1] + relativedelta(months=1)
    ts[new_index] = dat

def add_today_data(model, ts,  data, d, type='day'):
    _add_new_data(ts, data, type)  # 为原始序列添加数据
    # 为滞后序列添加新值
    d_ts = diff_ts(ts, d)
    model.add_today_data(d_ts[-1], type)

def forecast_next_day_data(model, type='day'):
    if model == None:
        raise ValueError('No model fit before')
    fc = model.forecast_next_day_value(type)
    return predict_diff_recover(fc, [12, 1])


df = open('flow1200.csv',)
data = pd.read_csv(df,encoding='utf-8', index_col='time')
data.index = pd.to_datetime(data.index)
ts_log = np.log(data)
ts_train = ts_log[:'2020-12-1']
ts_test = ts_log['2020-12-1':]

diffed_ts = diff_ts(ts_train)
forecast_list = []

for i, dta in enumerate(ts_test):
    if i%7 == 0:
        model = ARIMA(ts_train, order=(1, 0, 0))
    forecast_data = forecast_next_day_data(model, type='month')
    forecast_list.append(forecast_data)
    add_today_data(model, ts_train, dta, [12, 1], type='month')

predict_ts = pd.Series(data=forecast_list, index=data['2020-12-1':].index)
log_recover = np.exp(predict_ts)
original_ts = data['2020-12-1':]
