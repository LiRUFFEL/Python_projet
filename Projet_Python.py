# -*- coding: utf-8 -*-
"""
####################################################
                    Projet Python
             --Un robot investisseur
####################################################
commencé à 30/09/2020
"""

# pip install ccxt
# pip install shrimpy-python
# pip install pandas
# pip install plotly==4.1.0

###second week
import ccxt
from datetime import datetime
import plotly.graph_objects as go

binance = ccxt.binance()
trading_pair = 'BTC/TUSD'

candles = binance.fetch_ohlcv(trading_pair, '1d')
dates = []
open_data = []
high_data = []
low_data = []
close_data = []
for candle in candles:
    dates.append(datetime.fromtimestamp(candle[0] / 1000.0).strftime('%Y-%m-%d'))
    open_data.append(candle[1])
    high_data.append(candle[2])
    low_data.append(candle[3])
    close_data.append(candle[4])
    
fig = go.Figure(data=[go.Candlestick(x=dates,
                      open=open_data, high=high_data,
                      low=low_data, close=close_data)])
fig.write_html('first_figure.html', auto_open=True)
# pourquoi ca n'affiche pas le graph avec fig.show()
# est-ce que dash est equivalent à R shiny


###third week
#importing library
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## opening the csv file in 'w+' mode 
file = open('C:/Users/lizheng/Documents/PythonScripts/candles.csv', 'w+', newline ='')

# writing the data into the file 
with file:     
    write = csv.writer(file) 
    write.writerows(candles) 

##adding a new column named converted_time to the csv file

#CREATE dataframe with no column names by setting header=None
df=pd.read_csv('candles.csv', header=None)

#create dataframe with column names by wirting a header_list
header_list=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
df_new=pd.read_csv('candles.csv',names=header_list)

#adding a new column to df_new
#df_new["Converted_time"]=dates ##question ici : pourquoi pas df_new[Converted_time]


#write the new df to the csv file
df_new.to_csv("candles.csv", index=False)


###Descriptive statistics

#summarizing data
df_new.describe(include='all')#describe function does give the mean, std etc.??

#further details on the data
df_new.info()

#calculate the prix gap between open and close
#add a new column to the dataframe df_new
df_new["prix_gap"]=df_new["Open"]-df_new["Close"]

##plot with matplotlib.pyplot
df_new=df_new.cumsum()
plt.figure(); df_new.plot(); plt.legend(loc='best')
#le figure est que un ligne droit, bizzare,quel est le plt.figure()?plt()?
#quel est le différence avec plotly??


##plot avec plot
df_new.plot(x='Converted_time', y='prix_gap')#série chronologie
df_new.plot(x='Converted_time', y="Open") #we can see some trands from the plot
df_new.plot(x='Converted_time', y="Close") #we can see some trands from the plot

###est-ce que le prix a suivi une loi???lequel??

###'4th' week : algorithm for predicting the close price
#importing library for processing data
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#obtain the dataframe from the registered csv file
bitcoin_data=pd.read_csv('candles.csv', sep=',')

#create useful columns for the LSTM (long short term memory) model: bt_close_high_gap represents the gap between the closing price and price high for that day

kwargs={'bt_close_high_gap': lambda x: 2*(x['High']- x['Close'])/(x['High']-x['Low'])-1,
            'bt_volatility': lambda x: (x['High']- x['Low'])/(x['Open']) }
bitcoin_info = bitcoin_data.assign(**kwargs)

#select the columns from bitcoin_info to model
model_data=bitcoin_info[['Close','bt_close_high_gap','bt_volatility','Volume']]
model_data["dates"] = dates

#see the structure of the model_data : close and volume are not normalised
model_data.head()
model_data.tail()

# In time series models, we generally train on one period of time and then test on another separate period
# choose the split date 
split_date='2020-01-01'
# split the dateset 
training_set, test_set =model_data[model_data['dates']<split_date], model_data[model_data['dates']>=split_date]
# drop some columns that we don't need anymore
training_set=training_set.drop('dates', 1 )
test_set=test_set.drop( 'dates', 1 )
#normalise the data
norms_volume_close=['Volume', 'Close']

#LSTM training inputs and outputs 
window_len = 10
LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    for col in norms_volume_close:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1

#LSTM test inputs and outputs
LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    for col in norms_volume_close:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['Close'][window_len:].values/test_set['Close'][:-window_len].values)-1

#We would rather use numpy than pandas when all the datas are numeric : LSTM requires 3-D array
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)

# import the relevant Keras modules

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

#to build a model : firtstly, making an object of sequential model
#secondly add the lstm layers with parametrers
#input_shape: the shape of the training set
#Droupout layer is a type of regularization technique which is used to prevent overfitting


def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

#ramdom seed (Is it important that number of seed??)
np.random.seed(0)
#initialise the model
bt_model=build_model(LSTM_training_inputs,output_size=1,neurons=15)
# model output is next price normalised to 10th previous closing price
LSTM_training_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1
# train model on data
# note: bt_history contains information on the training error per epoch
bt_history = bt_model.fit(LSTM_training_inputs, LSTM_training_outputs, 
                            epochs=50, batch_size=1, verbose=2, shuffle=True)
# If things go as planning, we are expecting that the errors are decreasing with epoch
# we plot the errors with the epoches (training iterations)
fig, ax = plt.subplots()
ax.plot(bt_history.epoch, bt_history.history['loss'])
ax.set_title("training error")
plt.show()






