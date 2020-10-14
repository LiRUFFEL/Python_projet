# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:14:13 2020

@author: shade
"""

import plotly.graph_objects as go

import pandas as pd


#Importation de la dataframe
dt = pd.read_csv("C://Users//shade//OneDrive//Documents//M2_TIDE//Algorithmique_et_Python//Projet_S1//btc_aud.csv", sep = ",")


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

fig = go.Figure(data=[go.Candlestick(x=dt["dates"],
                      open=dt["open"], high=dt["high"],
                      low=dt["low"], close=dt["close"])])

fig.write_html("first_figure.html", auto_open = True)

fig_line = go.Figure()

fig_line.add_trace(go.Scatter(x=dt["dates"], y=dt["open"], name='Open Price',
                         line=dict(color='firebrick', width=1)))
fig_line.add_trace(go.Scatter(x=dt["dates"], y=dt["close"], name = 'Close price',
                         line=dict(color='royalblue', width=1)))
fig_line.add_trace(go.Scatter(x=dt["dates"], y=dt["high"], name = 'Highest price',
                         line=dict(color='black', width=1)))
fig_line.add_trace(go.Scatter(x=dt["dates"], y=dt["low"], name = 'Lowest price',
                         line=dict(color='magenta', width=1)))


fig_line.write_html("line_figure.html", auto_open = True)

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



fig_tx.write_html("tx_figure.html", auto_open = True)



#Représentation graphique des taux de variation open_t et prix_t
fig_tx_open = go.Figure()

fig_tx_open.add_trace(go.Scatter(x = dt["dates"], y=dt["tx_open_high"], name='Variation open_high',
                         line=dict(color='firebrick', width=1)))

fig_tx_open.add_trace(go.Scatter(x = dt["dates"], y=dt["tx_open_low"], name='Variation open_lox',
                         line=dict(color='royalblue', width=1)))

fig_tx_open.add_trace(go.Scatter(x = dt["dates"], y=dt["tx_open_close"], name='Variation open_close',
                         line=dict(color='magenta', width=1)))



fig_tx_open.write_html("tx_open_figure.html", auto_open = True)


#Représentation graphique des taux de variation prix_t et t-1
fig_ecart = go.Figure()

fig_ecart.add_trace(go.Scatter(x = dt["dates"], y=dt["open_to_median"], name='ecart à la mediane open',
                         line=dict(color='firebrick', width=1)))

fig_ecart.add_trace(go.Scatter(x = dt["dates"], y=dt["high_to_median"], name='ecart à la mediane high',
                         line=dict(color='royalblue', width=1)))

fig_ecart.add_trace(go.Scatter(x = dt["dates"], y=dt["low_to_median"], name='ecart à la mediane low',
                         line=dict(color='black', width=1)))

fig_ecart.add_trace(go.Scatter(x = dt["dates"], y=dt["close_to_median"], name='ecart à la mediane close',
                         line=dict(color='magenta', width=1)))



fig_ecart.write_html("ecart_figure.html", auto_open = True)










