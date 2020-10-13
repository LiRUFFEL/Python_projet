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

candles = binance.fetch_ohlcv(trading_pair, '1h')
dates = []
open_data = []
high_data = []
low_data = []
close_data = []
for candle in candles:
    dates.append(datetime.fromtimestamp(candle[0] / 1000.0).strftime('%Y-%m-%d %H:%M:%S.%f'))
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

## opening the csv file in 'w+' mode 
file = open('candles.csv', 'w+', newline ='')

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
df_new["Converted_time"]=dates ##question ici : pourquoi pas df_new[Converted_time]

#write the new df to the csv file
df_new.to_csv("candles.csv", index=False)








