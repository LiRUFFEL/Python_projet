# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:14:13 2020

@author: shade
"""

pip install statsmodels --upgrade
pip install tensorflow
import plotly.graph_objects as go

import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.graphics.tsaplots import plot_acf

from arch import arch_model

from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

import tensorflow_probability as tfp

from tensorflow_probability import sts
import plotly.express as px

#Importation de la dataframe
dt = pd.read_csv("C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//btc_aud_tout.csv", sep = ",")

dt.info()

dt.describe() #Statistiques descriptives des variables

#CALCUL DE TAUX DE VARIATION
#Calcul de var : prix_t et t-1

dt["tx_open"] = ((dt["open"] - (dt["open"].shift(periods = 1)))/(dt["open"].shift(periods = 1))) * 100

dt["tx_high"] = ((dt["high"] - (dt["high"].shift(periods = 1)))/(dt["high"].shift(periods = 1))) * 100

dt["tx_low"] = ((dt["low"] - (dt["low"].shift(periods = 1)))/(dt["low"].shift(periods = 1))) * 100

dt["tx_close"] = ((dt["close"] - (dt["close"].shift(periods = 1)))/(dt["close"].shift(periods = 1))) * 100

#Calcul de var : open_t et prix_t

dt["tx_open_high"] = ((dt["high"] - dt["open"])/dt["open"]) * 100

dt["tx_open_low"] = ((dt["low"] - dt["open"])/dt["open"]) * 100

dt["tx_open_close"] = ((dt["close"] - dt["open"])/dt["open"]) * 100

#Ecart à la médiane des prix

dt["open_to_median"] = dt["open"] - dt["open"].median()

dt["high_to_median"] = dt["high"] - dt["high"].median()

dt["low_to_median"] = dt["low"] - dt["low"].median()

dt["close_to_median"] = dt["close"] - dt["close"].median()



#Graphics

#Candlestick
fig = go.Figure(data=[go.Candlestick(x=dt["dates"],
                      open=dt["open"], high=dt["high"],
                      low=dt["low"], close=dt["close"])])

fig.write_html("C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//Candlestick.html", auto_open = True)

#Scatter y=open, x=open_t-1
scat=px.scatter(dt.iloc[50:], x="lag50_open", y="open")
scat.write_html('sc.html', auto_open = True)

#Representation graphique de chaque prix

fig_line = go.Figure()

fig_line.add_trace(go.Scatter(x=dt["dates"], y=dt["open"], name='Open Price',
                         line=dict(color='firebrick', width=1)))
fig_line.add_trace(go.Scatter(x=dt["dates"], y=dt["close"], name = 'Close price',
                         line=dict(color='royalblue', width=1)))
fig_line.add_trace(go.Scatter(x=dt["dates"], y=dt["high"], name = 'Highest price',
                         line=dict(color='black', width=1)))
fig_line.add_trace(go.Scatter(x=dt["dates"], y=dt["low"], name = 'Lowest price',
                         line=dict(color='magenta', width=1)))


fig_line.write_html("line_prices.html", auto_open = True)

#Représentation graphique des taux de variation prix_t et t-1
fig_tx = go.Figure()

fig_tx.add_trace(go.Scatter(x = dt["dates"], y=dt["tx_open"], name='Variation open',
                         line=dict(color='firebrick', width=1)))

fig_tx.add_trace(go.Scatter(x = dt["dates"], y=dt["tx_high"], name='Variation high',
                         line=dict(color='royalblue', width=1)))

fig_tx.add_trace(go.Scatter(x = dt["dates"], y=dt["tx_low"], name='Variation low',
                         line=dict(color='black', width=1)))

fig_tx.add_trace(go.Scatter(x = dt["dates"], y=dt["tx_close"], name='Variation close',
                         line=dict(color='magenta', width=1)))



fig_tx.write_html("C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//tx_variation.html", auto_open = True)



#Représentation graphique des taux de variation open_t et prix_t
fig_tx_open = go.Figure()

fig_tx_open.add_trace(go.Scatter(x = dt["dates"], y=dt["tx_open_high"], name='Variation open_high',
                         line=dict(color='firebrick', width=1)))

fig_tx_open.add_trace(go.Scatter(x = dt["dates"], y=dt["tx_open_low"], name='Variation open_lox',
                         line=dict(color='royalblue', width=1)))

fig_tx_open.add_trace(go.Scatter(x = dt["dates"], y=dt["tx_open_close"], name='Variation open_close',
                         line=dict(color='magenta', width=1)))



fig_tx_open.write_html("C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//tx_open_prices.html", auto_open = True)


#Représentation graphique des écarts à la médiane
fig_ecart = go.Figure()

fig_ecart.add_trace(go.Scatter(x = dt["dates"], y=dt["open_to_median"], name='ecart à la mediane open',
                         line=dict(color='firebrick', width=1)))

fig_ecart.add_trace(go.Scatter(x = dt["dates"], y=dt["high_to_median"], name='ecart à la mediane high',
                         line=dict(color='royalblue', width=1)))

fig_ecart.add_trace(go.Scatter(x = dt["dates"], y=dt["low_to_median"], name='ecart à la mediane low',
                         line=dict(color='black', width=1)))

fig_ecart.add_trace(go.Scatter(x = dt["dates"], y=dt["close_to_median"], name='ecart à la mediane close',
                         line=dict(color='magenta', width=1)))



fig_ecart.write_html("C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//ecart_mediane.html", auto_open = True)


#Représentation graphique de l'acf pour "open"
plot_acf(dt["open"])

#Représentation graphique du pacf
plot_pacf(dt["open"])


dt["lag1_open"]=dt["open"].shift(1)
dt["lag2_open"]=dt["open"].shift(2)
dt["lag50_open"]=dt["open"].shift(50)
dt["open_diff1"]=dt["open"]-dt["lag1_open"]
dt.head()


open_dt=pd.DataFrame(dt["open"])
open_dt.info()

############################### MODELISATION ##################################
n_train=round(len(dt)*0.7)

train, test=dt.loc[1:n_train, "open"], dt.loc[n_train:,"open"]


############# AR ###########
def AR(train_dt=train, test_dt=test, lag=1):
    model=AutoReg(train_dt, lags=lag)
    model_fit=model.fit()
    #print('Coefficients: %s' % model_fit.params)
    predictions_train=model_fit.predict(start=1, end=len(train_dt), dynamic=False)
    predictions_test=model_fit.predict(start=len(train_dt), end=len(train_dt)+len(test_dt)-1, dynamic=False)
    return(predictions_train, predictions_test)
    #rmse_train = sqrt(mean_squared_error(train_dt.iloc[:len(train_dt)-lag], predictions_train.iloc[:len(predictions_train)-lag]))
    #rmse_test = sqrt(mean_squared_error(test_dt.iloc[:len(test_dt)-lag], predictions_test.iloc[:len(predictions_test)-lag]))
    #return(predictions_train, predictions_test, rmse_train, rmse_test)


len(pred_train)-1, pred_test, rmse_train, rmse_test = AR()

RMSE_train=[]
RMSE_test=[]
for lag in range(1, 25):
    #pred_train, pred_test, rmse_train, rmse_test= AR(lag=4)
    pred_train, pred_test= AR(lag=lag)
    #RMSE_train.append(rmse_train)
    #RMSE_test.append(rmse_test)
    path_train="C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//predict_train"+str(lag)+".html"
    path_test="C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//predict_test"+str(lag)+".html"
    path_rmse_tr="C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//rmse_train"+str(lag)+".html"
    path_rmse_te="C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//rmse_test"+str(lag)+".html"
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
    
list(range(len(test)-1))


    rmse_te=go.Figure()
    rmse_te.add_trace(go.Scatter(x = range(len(rmse_test)-lag), y=rmse_test, name='RMSE test',
                         line=dict(color='firebrick', width=1)))
    rmse_te.write_html(path_rmse_te, auto_open = False)
    
    
    
    
    
    

fig_ecart = go.Figure()

fig_ecart.add_trace(go.Scatter(x = dt["dates"], y=dt["open_to_median"], name='ecart à la mediane open',
                         line=dict(color='firebrick', width=1)))

fig_ecart.add_trace(go.Scatter(x = dt["dates"], y=dt["high_to_median"], name='ecart à la mediane high',
                         line=dict(color='royalblue', width=1)))

fig_ecart.add_trace(go.Scatter(x = dt["dates"], y=dt["low_to_median"], name='ecart à la mediane low',
                         line=dict(color='black', width=1)))

fig_ecart.add_trace(go.Scatter(x = dt["dates"], y=dt["close_to_median"], name='ecart à la mediane close',
                         line=dict(color='magenta', width=1)))



fig_ecart.write_html("C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//ecart_mediane.html", auto_open = True)



model = arch_model(train, mean='Zero', vol='ARCH', p=1)

model_fit = model.fit()

print(model_fit.summary())
dt.loc[1:, "open_diff1"].mean()



fig_diff = go.Figure()

fig_diff.add_trace(go.Scatter(x = dt["dates"], y=dt["open_diff1"], name='Variation open_high',
                         line=dict(color='firebrick', width=1)))


fig_diff.write_html("tx_open_prices.html", auto_open = True)



















