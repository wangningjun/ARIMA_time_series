import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf, pacf
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.recurrent import LSTM


df = open('flow1200.csv')
data = pd.read_csv(df,encoding='utf-8', index_col='time')
data.index = pd.to_datetime(data.index)
percentage = 0.6
series = data.values
size = int(len(series) * 0.66)
train, test = series[0:size], series[size:len(series)]


def Autoc(Train):
    model = ARIMA(Train, order=(9,0,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()  # 残差分布图
    plt.show()
    residuals.plot(kind='kde')    # kde画密度图
    plt.show()
    print(residuals.describe())
    # plot the acf for the residuals
    acf_1 = acf(model_fit.resid)[1:20]
    plt.plot(acf_1)
    plt.show()
    test_df = pd.DataFrame([acf_1]).T
    test_df.columns = ["Pandas Autocorrelation"]
    test_df.index += 1
    test_df.plot(kind='bar')
    plt.show()
    # from the acf obtained from the residuals we concule that
    # there is still a nonlinear relationship among the residuals

"""
Hybrid Model
"""
window_size = 50
def make_model(window_size):
    model = Sequential()
    model.add(Dense(50, input_dim=window_size, init="uniform", activation="tanh"))
    model.add(Dense(25, init="uniform", activation="tanh"))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def ANN(Train):
    model = make_model(50)
    # lstm_model = make_lstm_model()
    min_max_scaler = preprocessing.MinMaxScaler()
    Train = np.array(Train).reshape(-1, 1)

    train_scaled = min_max_scaler.fit_transform(Train)

    train_X, train_Y = [], []
    for i in range(0, len(train_scaled) - window_size):
        train_X.append(train_scaled[i:i + window_size])
        train_Y.append(train_scaled[i + window_size])

    new_train_X, new_train_Y = [], []
    for i in train_X:
        new_train_X.append(i.reshape(-1))
    for i in train_Y:
        new_train_Y.append(i.reshape(-1))
    new_train_X = np.array(new_train_X)
    new_train_Y = np.array(new_train_Y)
    # new_train_X = np.reshape(new_train_X, (new_train_X.shape[0], new_train_X.shape[1], 1))
    model.fit(new_train_X, new_train_Y, nb_epoch=1000, batch_size=64, validation_split=.1)
    return model

def arima(Train,Test):
    """
    Arima Rolling Forecast
    """
    predicted1, resid_test = [], []
    history = Train.tolist()
    for t in range(len(Test)):
        model = ARIMA(history, order=(1, 0, 1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        resid_test.append(test[t] - output[0])
        predicted1.append(yhat)
        obs = test[t]
        obs = obs.tolist()
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs[0]))
    test_resid = []
    for i in resid_test:
        test_resid.append(i[0])
    error = mean_squared_error(test, predicted1)
    print('Test MSE: %.3f' % error)
    return predicted1,test_resid

def ANN_pre(model,test_resid):

    test_extended = train.tolist()[-1 * window_size:] + test_resid
    test_data = []
    for i in test_extended:
        try:
            test_data.append(i[0])
        except:
            test_data.append(i)
    test_data = np.array(test_data).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    test_scaled = min_max_scaler.fit_transform(test_data)
    test_X, test_Y = [], []
    for i in range(0, len(test_scaled) - window_size):
        test_X.append(test_scaled[i:i + window_size])
        test_Y.append(test_scaled[i + window_size])
    new_test_X, new_test_Y = [], []
    for i in test_X:
        new_test_X.append(i.reshape(-1))
    for i in test_Y:
        new_test_Y.append(i.reshape(-1))
    new_test_X = np.array(new_test_X)
    new_test_Y = np.array(new_test_Y)
    # new_test_X = np.reshape(new_test_X, (new_test_X.shape[0], new_test_X.shape[1], 1))
    predictions = model.predict(new_test_X)
    predictions_rescaled = min_max_scaler.inverse_transform(predictions)
    Y = pd.DataFrame(new_test_Y)
    pred = pd.DataFrame(predictions)
    error = mse(test_resid, predictions_rescaled)
    print('Test MSE: %.3f' % error)
    return pred


if __name__ == '__main__':
    predictions_ARIMA, test_resid = arima(train,test)
    model = ANN(train)
    predictions_ANN = ANN_pre(model,test_resid)
    pred_final = predictions_ARIMA + predictions_ANN + 18000
    error = mse(test, pred_final)
    print('Test MSE: %.3f' % error)
    Y = pd.DataFrame(test)
    pred = pd.DataFrame(pred_final)
    plt.plot(Y)
    plt.plot(pred, color='r')
    plt.show()
