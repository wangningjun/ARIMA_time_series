import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.seasonal import seasonal_decompose


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6

df = open('flow1200.csv',)
data = pd.read_csv(df,encoding='utf-8', index_col='time')
data.index = pd.to_datetime(data.index)   # 必须规范化日期索引

data_log = np.log(data)
data_log.dropna(inplace=True)
# data_log = data_log.values
# 分解decomposing
decomposition = seasonal_decompose(data_log, model="additive")

trend = decomposition.trend  # 趋势
seasonal = decomposition.seasonal  # 季节性
residual = decomposition.resid  # 剩余的

plt.subplot(411)
plt.plot(data_log,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonarity')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residual')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
