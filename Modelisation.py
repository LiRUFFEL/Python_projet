# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:11:55 2020

@author: shade
"""
#import : import module
#from ... import : import function from module
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import plotly.express as px
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


#Importation de la dataframe
dt = pd.read_csv("C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//btc_aud_tout.csv", sep = ",")
dt["centré"]=dt["open"]-dt["open"].mean()
dt["lag1_centré"]=dt["centré"].shift(1)
dt["diff1_centré"]=dt["centré"]-dt["centré"].shift(1)
dt["diff1"]=dt["open"]-dt["open"].shift(1)
dt["taux"] = (dt["close"] - dt["close"].shift(1))/dt["close"].shift(1)
#Représentation graphique de l'acf pour "open"
plot_acf(dt["open"])
plot_acf(dt["centré"])
plot_acf(dt["taux"].iloc[1:])
plot_acf(dt["diff1"].iloc[1:])
#Représentation graphique du pacf pour "open"
plot_pacf(dt["open"])
plot_pacf(dt["centré"])
plot_pacf(dt["diff1_centré"].iloc[1:])
plot_pacf(dt["diff1"].iloc[1:])

#Test de Dickey Fuller
test_fuller=adfuller(dt["diff1_centré"].iloc[1:])
print('p-value: %f' % test_fuller[1])
#Test de Ljung sur "diff1_centré"
test_box=acorr_ljungbox(dt.loc[1:,"diff1_centré"], return_df=True)
test_box.sort_values("lb_pvalue", ascending=True)#Rejet de H0:présente d'autocorrelation


#Graphique line
fig_line = go.Figure()

fig_line.add_trace(go.Scatter(x=dt["dates"], y=dt["diff1_centré"], name='diff1_centré',
                         line=dict(color='firebrick', width=1)))
fig_line.add_trace(go.Scatter(x=dt["dates"], y=dt["diff1"], name = 'diff1',
                         line=dict(color='royalblue', width=1)))
fig_line.write_html("line_prices.html", auto_open = True)



#Chronologic train-test split
n_train=round(len(dt)*0.7)

train, test=dt.loc[1:n_train, "diff1_centré"], dt.loc[n_train+1:,"diff1_centré"] # .loc inclut les limites inférieurs et supérieures dans la data frame
len(train)
len(test)



############# AR ###########
def AR(train_dt=train, test_dt=test, lag=1):
    model=AutoReg(train_dt, lags=lag)
    model_fit=model.fit()
    #print('Coefficients: %s' % model_fit.params)
    predictions_train=model_fit.predict(start=lag+1, end=len(train_dt)-1, dynamic=False)
    predictions_test=model_fit.predict(start=len(train_dt), end=len(train_dt)+len(test_dt)-1, dynamic=False)
    
    return(predictions_train, predictions_test)
    #rmse_train = sqrt(mean_squared_error(train_dt.iloc[:len(train_dt)-lag], predictions_train.iloc[:len(predictions_train)-lag]))
    #rmse_test = sqrt(mean_squared_error(test_dt.iloc[:len(test_dt)-lag], predictions_test.iloc[:len(predictions_test)-lag]))
    #return(predictions_train, predictions_test, rmse_train, rmse_test)


len(dt)-1, pred_test, rmse_train, rmse_test = AR()

RMSE_train=[]
RMSE_test=[]
for lag in range(1, 4):
    #pred_train, pred_test, rmse_train, rmse_test= AR(lag=4)
    pred_train, pred_test= AR(lag=lag)
    #RMSE_train.append(rmse_train)
    #RMSE_test.append(rmse_test)
    path_train="C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//Graphs//predict_train"+str(lag)+".html"
    path_test="C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//Graphs//predict_test"+str(lag)+".html"
    path_rmse_tr="C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//Graphs//rmse_train"+str(lag)+".html"
    path_rmse_te="C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//Graphs//rmse_test"+str(lag)+".html"
    predict_test= go.Figure()
    predict_test.add_trace(go.Scatter(x = list(range(len(pred_test)-lag)), y=pred_test.iloc[:len(pred_test)-lag], name='Predictions test',
                         line=dict(color='firebrick', width=1)))
    predict_test.add_trace(go.Scatter(x = list(range(len(test)-lag)), y=test.iloc[:len(test)-lag], name='Vraies valeurs test',
                         line=dict(color='royalblue', width=1)))
    predict_test.write_html(path_test, auto_open = False)
    predict_train=go.Figure()
    predict_train.add_trace(go.Scatter(x = list(range(len(pred_train)-lag)), y=pred_train.iloc[:len(pred_train)-lag], name='Predictions train',
                         line=dict(color='firebrick', width=1)))
    predict_train.add_trace(go.Scatter(x = list(range(len(train)-lag)), y=train.iloc[:len(train)-lag], name='Vraies valeurs train',
                         line=dict(color='royalblue', width=1)))
    predict_train.write_html(path_train, auto_open = False)



############# ARIMA ###########

mod_arima = ARIMA(train, order=(1, 0, 1))
mod_arima_fit = mod_arima.fit()
type(predictions_train)=mod_arima_fit.predict(start=2, end=len(train)-1, dynamic=False)
predictions_test=mod_arima_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    
    predict_test= go.Figure()
    predict_test.add_trace(go.Scatter(x = list(range(len(predictions_test))), y=predictions_test, name='Predictions test',
                         line=dict(color='firebrick', width=1)))
    predict_test.add_trace(go.Scatter(x = list(range(len(test))), y=test, name='Vraies valeurs test',
                         line=dict(color='royalblue', width=1)))
    predict_test.write_html(path_test, auto_open = True)
    predict_train=go.Figure()




###################### LSTM ######################

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import LeakyReLU
from sklearn.metrics import mean_squared_error as mse
from sklearn import metrics


signal = dt["taux"][1:]#Taux de croissance du prix close

def mise_en_forme(raw_data, lag_max):
    x, y = [], []
    for i in range(len(raw_data)):
        end_ix = i + lag_max
        if end_ix > len(raw_data)-1 :
            print(end_ix)
            break
        seq_x, seq_y = raw_data[i:end_ix], raw_data[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

n_steps = 3
x, y = mise_en_forme(signal, n_steps)

n_features = 1
x = x.reshape((x.shape[0], x.shape[1], n_features))

#Train_test_split chronologique
n_train=round(len(x)*0.7)
x_train = x[:n_train]
x_test = x[n_train:]
y_train = y[:n_train]
y_test = y[n_train:]

# Construction du modele Vanilla LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(n_steps, n_features)))
model.add(LeakyReLU())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(x_train, y_train, epochs=200, verbose=0)

#Prediction
pred_train = model.predict(x_train, verbose=0)

pred_train_graph = pred_train.reshape(len(pred_train))

mse_train = sqrt(mse(y_train, pred_train_graph))

line_train = go.Figure()

line_train.add_trace(go.Scatter(x=list(range(len(x_train))), y=y_train, name='vraies',
                         line=dict(color='firebrick', width=1)))
line_train.add_trace(go.Scatter(x=list(range(len(x_train))), y=pred_train_graph, name = 'pred',
                         line=dict(color='royalblue', width=1)))
line_train.write_html("C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//Graphs//pred_train_close_taux.html", auto_open = True)

line_train_sub = make_subplots(rows=3, cols=1)

line_train_sub.append_trace(go.Scatter(x=list(range(len(x_train))), y=y_train, name='vraies',
                         line=dict(color='firebrick', width=1)), row=1, col=1)
line_train_sub.append_trace(go.Scatter(x=list(range(len(x_train))), y=pred_train_graph, name = 'pred',
                         line=dict(color='royalblue', width=1)), row=1, col=1)
line_train_sub.append_trace(go.Scatter(x=list(range(len(x_train))), y=y_train, name='vraies',
                         line=dict(color='firebrick', width=1)), row=2, col=1)
line_train_sub.append_trace(go.Scatter(x=list(range(len(x_train))), y=pred_train_graph, name = 'pred',
                         line=dict(color='royalblue', width=1)), row=3, col=1)
line_train_sub.write_html("C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//Graphs//pred_train_close_taux_sub.html", auto_open = True)


pred_test = model.predict(x_test, verbose=0)


pred_test_graph = pred_test.reshape(len(pred_test))

mse_test = sqrt(mse(y_test, pred_test_graph))

line_test = go.Figure()

line_test.add_trace(go.Scatter(x=list(range(len(x_test))), y=y_test, name='vraies',
                         line=dict(color='firebrick', width=1)))
line_test.add_trace(go.Scatter(x=list(range(len(x_test))), y=pred_test_graph, name = 'pred',
                         line=dict(color='royalblue', width=1)))
line_test.write_html("C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//Graphs//pred_test_close_taux.html", auto_open = True)

line_test_sub = make_subplots(rows=3, cols=1)

line_test_sub.append_trace(go.Scatter(x=list(range(len(x_test))), y=y_test, name='vraies',
                         line=dict(color='firebrick', width=1)), row=1, col=1)
line_test_sub.append_trace(go.Scatter(x=list(range(len(x_test))), y=pred_test_graph, name = 'pred',
                         line=dict(color='royalblue', width=1)), row=1, col=1)
line_test_sub.append_trace(go.Scatter(x=list(range(len(x_test))), y=y_test, name='vraies',
                         line=dict(color='firebrick', width=1)), row=2, col=1)
line_test_sub.append_trace(go.Scatter(x=list(range(len(x_test))), y=pred_test_graph, name = 'pred',
                         line=dict(color='royalblue', width=1)), row=3, col=1)
line_test_sub.write_html("C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//Graphs//pred_test_close_taux_sub.html", auto_open = True)


def action(taux_croissance):
    #Cette fonction détermine si on devrait vendre ou acheter du btc selon le taux de croissance
    if taux_croissance < 0:
        faire = "acheter"
    elif taux_croissance > 0:
        faire = 'vendre'
    else:
        faire = 'passer'
    return faire

def action_sequence(taux):
    #Cette fonction détermine si on devrait vendre ou acheter du btc selon une liste de taux de croissance
    act_list = []
    for taux_croissance in taux:
        if taux_croissance < 0:
            faire = "acheter"
        elif taux_croissance > 0:
            faire = 'vendre'
        else:
            faire = 'passer'
        act_list.append(faire)
    return act_list

actions_test = action_sequence(pred_test_graph)  

actions_vrai_test = action_sequence(y_test)  
    
metrics.accuracy_score(actions_vrai_test, actions_test)

def stock(stock_aud, stock_btc, faire, btc_aud, proportion_stock = 0.25):
    #Cette fonction permet de déterminer le stock de cash après achat ou vente de btc
    #proportion_stock du stock de cash qu'on souhaite acheter ou vendre
    if faire == "acheter":
        stock_aud_finale = stock_aud - stock_aud*proportion_stock
        stock_btc_finale = stock_btc + (stock_aud*proportion_stock)/btc_aud
    elif faire == "vendre":
        stock_aud_finale = stock_aud + stock_btc*proportion_stock*btc_aud
        stock_btc_finale = stock_btc - stock_btc*proportion_stock
    else:
        stock_aud_finale = stock_aud 
        stock_btc_finale = stock_btc
    return stock_aud_finale, stock_btc_finale

cours = dt["close"].iloc[n_train+1:len(dt["close"])-n_steps]#cours btc/aud à la cloture

stock_aud = 1000000
stock_btc = 3
stock_aud_test = []
stock_btc_test = []
stock_aud_vrai_test = []
stock_btc_vrai_test = []
for i in range(len(actions_test)):
    aud,btc=stock(stock_aud, stock_btc, actions_test[i], cours.iloc[i], proportion_stock = 0.25)
    stock_aud_test.append(aud)
    stock_btc_test.append(btc)

for i in range(len(actions_vrai_test)):
    aud,btc=stock(stock_aud, stock_btc, actions_vrai_test[i], cours.iloc[i], proportion_stock = 0.25)
    stock_aud_vrai_test.append(aud)
    stock_btc_vrai_test.append(btc)

sqrt(mse(stock_aud_test, stock_aud_vrai_test))   
sqrt(mse(stock_btc_test, stock_btc_vrai_test))    
    
stock_line = go.Figure()

stock_line.add_trace(go.Scatter(x=list(range(len(x_test))), y=stock_aud_vrai_test, name='vraies',
                         line=dict(color='firebrick', width=1)))
stock_line.add_trace(go.Scatter(x=list(range(len(x_test))), y=stock_aud_test, name = 'pred',
                         line=dict(color='royalblue', width=1)))
stock_line.write_html("C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//Graphs//stock_evolution.html", auto_open = True)





    





















































