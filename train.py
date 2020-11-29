import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
# from tensorflow.keras.metrics import MeanAbsolutePercentageError
from sklearn.preprocessing import MinMaxScaler
import talib


data = pd.read_csv('EURUSD.csv')

scaler = MinMaxScaler()
scaler_y = MinMaxScaler()
data['ROC'] = talib.RSI(data.Close,timeperiod=10)
data['ROCP'] = talib.ROCP(data.Close, timeperiod=10)
data['EMA-9'] = talib.EMA(data.Close,timeperiod=9)
data['EMA-28']= talib.EMA(data.Close,timeperiod=28)
data['SAR'] = talib.SAR(data.High, data.Low, acceleration=0, maximum=0.2)
data['upperband'], data['middleband'], data['lowerband'] = talib.BBANDS(data.Close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
data['Real'] = talib.TRANGE(data.High, data.Low, data.Close)
# data = scaler.fit_transform(data)
# data = pd.DataFrame(data,columns=['High','Low','Close'])
data['mean']=(data.High+data.Low)/2
data.drop(labels=['Open','Gmt time','Volume','High','Low'],axis=1,inplace=True)
data.dropna(inplace=True)
lable= data.Close.shift(periods=-17)
data.replace([np.inf, -np.inf],0,inplace=True)
data = scaler.fit_transform(data)
data = pd.DataFrame(data)
y = [i for i in lable if str(i) != 'nan']
y=np.array(y,dtype=np.float)
y = scaler_y.fit_transform(y.reshape(-1,1))
# print(data.head())
x=[]
for i in range(16,len(data)+1):
    x.append(data.iloc[i-16:i].values)
x = x[:len(y)]
x = np.array(x,dtype=np.float)
print(np.any(np.isnan(x)))
print(np.any(np.isinf(x)))
print(np.any(np.isnan(y)))
print(np.any(np.isinf(y)))

X_train_merged, X_val_and_test, Y_train_merged, Y_val_and_test = train_test_split(x, y, test_size=0.4)

X_val_merged, X_test_merged, Y_val_merged, Y_test_merged = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
print(x.shape)
print(y.shape)

model = Sequential()
model.add(LSTM(16,return_sequences=True,activation='relu',input_shape=(16,11)))
model.add(Dropout(0.3))
model.add(LSTM(16,return_sequences=True))
model.add(Dropout(0.3))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(optimizer='adam',
                loss='mse',
                metrics=['mae','mse'])
model.fit(x=X_train_merged, y=Y_train_merged, epochs=25,batch_size=500,validation_data=(X_val_merged,Y_val_merged))

model.save('models/model3.h5')

import matplotlib.pyplot as plt


out = model.predict(X_test_merged)
o = []
for i in out:
    o.append(i[0])

plt.figure(figsize=(20,20))
plt.plot(range(1,101),o[:100])
plt.plot(range(1,101),Y_test_merged[:100])
# plt.plot(Y_test_merged)

plt.legend(["Pre", "Act"])

plt.show()