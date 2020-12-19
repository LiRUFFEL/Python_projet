# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:50:15 2020

@author: Assa Li Shade
"""

########################################## PROJET : Algorithme : Investissement d'action  #################################

###########################################################################################################################
###########################################Partie A : ARIMA #######################################################################
###########################################################################################################################

#On essay de voir l'application de ARIMA
#pip install plotly==4.11.0
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import itertools
from itertools import product
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import math
import warnings
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# I - Aperçu du Data Frame
# II - Analyse du data frame
# II - Estimation 

######################################### I- Aperçu du Data Frame##########################################################

###########################################################################################################################


#Importation des données du data-frame df_coin 

df_coin=pd.read_csv('btc_aud_tout.csv')

#Affichage du type de df_coin 

type(df_coin)

#Affichage des colonnes en utilisant l'attribut columns

df_coin.columns 
print(df_coin)

#Dimension du data frame 

df_coin.shape

#Affichage des 7 premières lignes en utilisant la méthode head 

df_coin.head(7)

#Affichage des 7 dernières lignes en utilisant la méthode tail

df_coin.tail(7)


#Affichage des informations globales du data frame (nbr de colonnes, nbr de lignes, nbr de valeurs manquantes..)

df_coin.info()

#Type de chaque colonne

df_coin.dtypes

#Statistiques globales sur toutes les variables numériques : méthode describe

df_coin.describe()

###########################################################################################################################




######################################### II-Analyse graphique du data frame ##############################################

###########################################################################################################################


# Un moyen  d'investir est d'utiliser un actif financier

#On s'interessera donc a la serie financiere qui correspond au prix a la clôture d'un actif financier contenue dans la librairie CCXT de python.

#On représente graphiquement le cours à la clôture en fonction du temps.


np.random.seed(1)

N = 100
random_x = df_coin.dates

# traçage de la figure

fig = go.Figure()

fig.add_trace(go.Scatter(x=random_x, y=df_coin.close,
                    mode='lines',
                    name='close price'))


fig.update_layout(title=" Trajectoire représentant le prix à la cloture selon le temps")

#Commentaire:
    
#On observe que le cours de clôture a tendance a augmenté entre Avril 2019 jusqu'a fin Juin 2019 (le pic est atteint juste avant Juillet 2019).
#De juillet 2019 jusqu'à Octobre 2020, on observe des fluctuations; à certains moments il y a des augmentations du prix, et à d'autres moments des baisses du prix.
# On observe également un autre pic ; on observe une forte baisse du prix au mois de Mars 2020
#On peut tracer l'ACF d'une telle série chronologique même si on sait à l'avance qu'étant donné que la série est non-stationnaire
#l'autocorrélation empirique estimera très mal l'autocorrélation. Par conséquent, on interprétera mal l'ACF. 

#Le prix à la clôture n'est pas stationnaire car on pouvait observer un phénomène de tendance.
#Représentons le graphe de moyenne mobile, de l'ec-type mobile et de la série et faisons un test de D-F pour confirmer cela.

def stationarity(timeseries):
    
    # Statistiques mobiles
    rolling_mean = timeseries.rolling(window=25).mean() #moyenne mobile
    rolling_std = timeseries.rolling(window = 25).std() #ecartype-mobile
    
    # tracé statistiques mobiles
    originale = plt.plot(timeseries, color='blue', label='Données brutes')
    mean = plt.plot(rolling_mean, color='red', label='Moyenne Mobile') #tendance estimé par MA
    ec_type = plt.plot(rolling_std, color = 'black', label = 'Ecart-type mobile')
    plt.legend(loc='best') 
    plt.title('Moyenne et Ecartype mobile')
    plt.show(block=False)
    
    # Test Dickey–Fuller : 
    result = adfuller(timeseries)
    print('Statistiques ADF : {}'.format(result[0]))
    print('p-value : {}'.format(result[1]))
    print('Valeurs Critiques :')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))



stationarity(df_coin["close"]) 


#Graphique Moyenne et Ecartype Mobile
#On observe que la moyenne mobile n'est pas constante, elle augmente ou baisse avec le temps, bien que l'ec-type mobile reste plus ou moins constant

#Test de Dickey-Fuller
#Le test de Dickey-Fuller est définit comme suit; H0 :" La série n'est pas stationnaire" vs H1= "La série est stationnaire"
#On obtient une p-value de 16% donc on rejete H0 

#Ce qui nous confirme que la série n'est pas stationnaire.
#l'ACF de la série chronologique


 
plot_acf (df_coin.close, lags = 90)

# Nous voyons bien toutes les limites de l'ACF. En effet, nous ne pouvons pas conclure sur le fait que les variables de la série soient indépendantes dès lors qu'il y a un phénomène de tendance qui apparaît


###########################################################################################################################


# L'objectif de notre projet sera surtout de savoir quel est le moment opportun pour investir dans une telle action afin de réaliser un bénéfice.
#Essayons de mesurer la volatilité de l'actif c'est-à-dire le risque ( à la hausse ou à la perte ) que supporte un investisseur en détenant un tel actif.
# Rappel : volatilité d'une action = propension à subir des mouvements de prix plus ou moins prononcés.
# A priori,si on suppose qu'on achète l'actif en Mars 2019, on pourrait penser que le meilleur moment pour vendre notre actif serait aux alentours de fin Juin 2019 
#car le prix de clôture atteint son maximum, par conséquent on pourrait  réalisé un grand bénéfice ( de 15M de dollars ).
# Et à contrario, le moment le moins opportun durant la période observé, serait en Mars 2020  car ferait seulement un bénéfice de 2M .
#Voyons cela de plus près...

#On calcule les log-rendements de la série entre le prix_t et t-1

logR=np.log(df_coin["close"]/df_coin["close"].shift(1))*100

#On représente graphiquement le log-rendement en fonction du temps


np.random.seed(1)

N = 100
random_x = df_coin.dates

# traçage de la figure

fig = go.Figure()
fig.add_trace(go.Scatter(x=random_x, y=logR,
                    mode='lines',
                    name='log rendement'))


fig.update_layout(title=" Trajectoire représentant le log-rendement selon le temps")


#Commentaire : 
#Ce qu'on observe : 
#Le log-rendement tourne plus ou moins autour de 5%/7%, l'actif est globalement peu volatile.
#Cependant, on observe des pics ; il y a une très forte volatilité de l'actif autour du mois de Mars 2020 ( environ 19%), donc le risque qu'encourt l'investisseur est important à ce moment.
#On observe également un autre pic en fin Juin 2019, car le cours atteint son maximum le 26 Juin 2020 et ensuite chute le 27 Juin 2020.
#Cependant, on se rend compte que le signal qu'on observe nous apporte pas énormément d'informations .
#En effet, la série des log-rendement étant très bruitée, on ne peut pas isoler des tendances pour avoir une stratégie de vente.
#Nous allons donc par la suite, travailler uniquement avec le prix de clôture ( close-price)

######################################### III- Estimation paramétrique  ###################################################

###########################################################################################################################

#A ) Estimation paramétrique du price-close avec différents processus.

#  ESTIMATION AVEC L'ALGORITHME ARIMA 

#Pq ce choix d'algortihme ? Le modèle ARIMA permet de faire des prédictions même si la série est non -stationnaire

#  ARIMA est la combinaison de trois termes : 
# le terme autorégressif (AR), le terme de différenciation (I) et le terme de moyennes mobiles (MA)
# ARIMA (p,d,q):
# p est le nombre de termes auto-régressifs
# d est le nombre de différences
# q est le nombre de moyennes mobiles

# order=() ce sont respectivement les coefficients AR, I et MA je test le modèle avec p=1 , d=1 et q=1 
mdl = sm.tsa.statespace.SARIMAX(df_coin["close"],order=(1, 1, 1))
res = mdl.fit()
print(res.summary())

#Observations : On constate que 3 coefficients ont été estimés :
# 1 coefficient du terme auto-régressif : 0.3990
# le terme de moyenne mobile : -0.4656
# et la variance du bruit :  9640.3804   

#Les équartypes estimés des estimateurs de ces coefficients sont les suivants 
# pour le terme AR 0.039, 
#pour le terme MA 0.037, 
#pour la variance du bruit 28.790

#Les tests de student nous informent que les coefficients sont tous non-nuls donc qu'ils sont signifcatifs.


#Graphiques de la modélisation
res.plot_diagnostics(figsize=(16, 10))
plt.tight_layout()
plt.show()

y = pd.DataFrame(df_coin["close"])



# adapter le modèle aux données
res = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                enforce_stationarity=True,
                                enforce_invertibility=True).fit()
 
# Limites de la prédiction
pred = res.get_prediction(start = 430,
                          dynamic = False, 
                          full_results=True)

# Graphe de la prédiction et des données
fig = plt.figure(figsize=(19, 7))
ax = fig.add_subplot(111)
ax.plot(y[0:],color='#006699',linewidth = 3, label='Observation');
pred.predicted_mean.plot(ax=ax,linewidth = 3, linestyle='-', label='Prediction', alpha=.7, color='#ff5318', fontsize=18);
ax.set_xlabel('temps', fontsize=18)
ax.set_ylabel('logR', fontsize=18)
plt.legend(loc='upper left', prop={'size': 20})
plt.title('Prediction ARIMA', fontsize=22, fontweight="bold")
plt.show()



# Précision du modèle en calculant le RMSE
rmse = math.sqrt(((pred.predicted_mean.values.reshape(-1, 1) - y[430:].values) ** 2).mean())
print('rmse = '+ str(rmse))

#rmse = 99.48

#Conclusion : 
# On peut remarquer que l'estimation obtenue en utilisant l'algorithme ARIMA est très satisfaisante.
# En effet, la courbe rouge ( estimation par ARIMA) superpose presque la courbe bleu ( trajectoire des données brutes).
#On aimerait savoir si on peut améliorer cette modélisation.
# Quel est le meilleur modèle pour modéliser la trajectoire de notre prix de clôture. 
#Est-il préférable de la modéliser par un ARIMA(1,1,1)? ou par un ARIMA(1,1,3)? ou par un ARIMA(2,2,5)... ?
#Nous allons considérer une famille M de tous les ARIMA, jusqu'à un certain ordre et on va essayer de trouver le meilleur modèle
#pour modéliser cette série de données brutes en utilisant un critère BIC.
#Pq le BIC et pas l'AIC ? Les deux permettent de minimiser les erreurs de prédictions mais l'AIC a un défaut celui de la suparamétrisation.


#SELECTION DE MODELE

# Initialisation 
p= range(0, 6)
q= range(0, 5)
d=1
parameters = product(p,q)
parameters_list = list(parameters)
len(parameters_list)

# Selection de modèle
results = []
best_bic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        modele=sm.tsa.statespace.SARIMAX(df_coin["close"], order=(param[0], d, param[1])).fit()
    except ValueError:
        print('wrong parameters:', param)
        continue
    bic = modele.bic
    if bic < best_bic:
        best_modele = modele
        best_bic = bic
        best_param = param
    results.append([param, modele.bic])


# Affichage des résultats et du meilleur modèle
result_table = pd.DataFrame(results) #on met les résultats dans un dataframe
result_table.columns = ['parameters', 'bic']  #on souhaite uniquement afficher les colomnes des parametres et du critere BIC
print(result_table.sort_values(by = 'bic', ascending=True).head()) # Affichage des premiers meilleurs modèles (qui ont le + petit BIC)
print(best_modele.summary()) #affichage du meilleur modèle

#Le meilleur modèle pour modéliser le prix de clôture est le modèle ARIMA(2, 1, 0).

#Analyse des résidus du modèle ARIMA(2,1,0) pour s'assurer que c'est un bon modèle  

plt.figure(figsize=(15,7))
plt.subplot(300)
best_modele.resid[13:].plot()
plt.ylabel(u'Residuals')
ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_modele.resid[13:].values.squeeze(), lags=48, ax=ax)

print("Dickey–Fuller test:: p=%f" % sm.tsa.stattools.adfuller(best_modele.resid[13:])[1])

plt.tight_layout()
plt.show()

#On accepte que le modèle est bon pour modéliser la trajectoire du prix de clôture.


###########################################################################################################################
###########################################Partie B : LSTM #######################################################################
###########################################################################################################################

#on veux essayer LSTM pour traiter les donnes de cryptomonais
#on veux tenter cette méthode car elle a eu de bonnes performances pour estimer et prédire dans le monde de cryptomonais
#on veux voir aussi btc_usd crypotomonais par jour
#on a 4 parties pour cette modèle
#I- Aperçu du Data Frame et ajout des variables pour modèle LSTM
#II-Préparation des données
#III-Preparation de modèle LSTM
#IV- Plotting and evaluation

######################################### I- Aperçu du Data Frame##########################################################

###########################################################################################################################
bitcoin_data=pd.read_csv('btc_usd_jour.csv', sep=',')

#create some more features for the LSTM (long short term memory) model: 
#bt_close_high_gap represents the gap between the closing price and price high for that day

kwargs={'btc_close_high_gap': lambda x: (x['High']- x['Close'])/(x['High']-x['Low'])-1,
            'btc_volatility': lambda x: (x['High']- x['Low'])/(x['Open']) }
bitcoin_info = bitcoin_data.assign(**kwargs) #Assign new columns to a DataFrame with function assign()

#select the columns from bitcoin_info to model
model_data=bitcoin_info[['Close','btc_close_high_gap','btc_volatility','Volume','dates']]
type(model_data['dates'].iloc[1]) #the value of column 'dates' is a string

#convert the date string to a correct date format
model_data['dates']=pd.to_datetime(model_data['dates'])
#type(model_data['dates'].iloc[1]) check again the type of the value in the column 'dates'

#see the structure of the model_data : close and volume are not normalised
model_data.head()
model_data.tail()
model_data['btc_close_high_gap'].max()
model_data['btc_close_high_gap'].min()

######################################### II-Préparation des données ##############################################

###########################################################################################################################
# In time series models, we generally train on one period of time and then test on another separate period
# choose the split date 
split_date='2020-01-01'
# split the dateset 
training_set, test_set =model_data[model_data['dates']<split_date], model_data[model_data['dates']>=split_date]
# drop column 'dates' that we don't need for modeling 
training_set=training_set.drop('dates', 1 )
test_set=test_set.drop( 'dates', 1 )
#construct the training inputs and outputs and normalise the data
norms_volume_close=['Volume', 'Close']
lag = 7
LSTM_training_inputs = []
for i in range(len(training_set)-lag):
    temp_set = training_set[i:(i+lag)].copy()
    for col in norms_volume_close:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1#we normalise the close price locally under its window
    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['Close'][lag:].values/training_set['Close'][:-lag].values)-1

#LSTM test inputs and outputs
LSTM_test_inputs = []
for i in range(len(test_set)-lag):
    temp_set = test_set[i:(i+lag)].copy()
    for col in norms_volume_close:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['Close'][lag:].values/test_set['Close'][:-lag].values)-1

#We would rather use numpy than pandas when all the datas are numeric : LSTM requires 3-D array
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)


######################################### III- Preparation de modèle LSTM ##########################################################

###########################################################################################################################

# import the relevant Keras modules

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

#to build a model : firtstly, making an object of sequential model
#secondly add the lstm layers with parametrers
#input_shape: the shape of the training set
#Droupout layer is a type of regularization technique which is used to prevent overfitting

#generally people define a build_model function for conducting the LSTM model with its different layers

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.15, loss="mae", optimizer="adam"):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

#ramdom seed 
np.random.seed(0)
#initialise the model
btc_model=build_model(LSTM_training_inputs,output_size=1,neurons=20)
# model output is next price normalised to 7th previous closing price
LSTM_training_outputs = (training_set['Close'][lag:].values/training_set['Close'][:-lag].values)-1
# train model on data
# note: btc_history contains information on the training error per epoch
btc_history = btc_model.fit(LSTM_training_inputs, LSTM_training_outputs, 
                            epochs=60, batch_size=1, verbose=2, shuffle=True)
# If things go as planning, we are expecting that the errors are decreasing with epoch
# we plot the errors with the epoches (training iterations)
fig, ax = plt.subplots()
ax.plot(btc_history.epoch, btc_history.history['loss'])
ax.set_title("training error")
plt.show()

#calculate MAE (mean absolute error) : one of the many metrics for summarizing and-- 
#assessing the quality of a machine learning model
MAE=np.mean(np.abs((np.transpose(btc_model.predict(LSTM_training_inputs))+1)-
            (training_set['Close'].values[lag:])/(training_set['Close'].values[:-lag])))
print(MAE)

######################################### IV- Plotting and evaluation ##########################################################

###########################################################################################################################
########################## Plot the actual and predicted price with training data ##########################

import datetime
#take the predicted price back to its original scale to plot
price_pred=((np.transpose(btc_model.predict(LSTM_training_inputs))+1) * training_set['Close'].values[:-lag])[0]
fig, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(model_data[model_data['dates']< split_date]['dates'][lag:], training_set['Close'][lag:], label='Actual')
ax1.plot(model_data[model_data['dates']< split_date]['dates'][lag:], price_pred, label='Predicted')
ax1.annotate('MAE: %.4f'%MAE, xy=(0.75, 0.75),  xycoords='axes fraction')#The point (x, y) to annotate and the coordinate system is determined by xycoords.

ax1.legend() # even though we have already configured the label in ax1.plot, we must to give ax1.legend to show the legends in the plot.

plt.show()
#Some comments for the results obtained from above: we can see that the taining errors are quite low, and the MAE too. if we use hundreds of neurons, 
#the traning error will get near zero. but in machine learning, it's not scientific to use the same data for training and prediction. 
#So to verify the quality of the model, we need to test it with our test dataset.

############################ Calculate and plot the performance of test dataset ############################

yhat=btc_model.predict(LSTM_test_inputs)
yhat.shape #(299,1)
#yhat contains the predicted values based on the model we constructed, the values od it are normalised. 
#we need to convert the values of yhat into the "non-normalised" values to compare with the true close price then.
y_converted=((np.transpose(yhat)+1)*test_set['Close'].values[:-lag])[0]
np.transpose(yhat).shape #(1,299) still a 2-D array
y_converted.shape #(299,) a 1-D array that is allowed by matplotlib

#calculate the MAE
# MAE_test=np.mean(np.abs(y_converted-test_set['Close'].values[window_len:])) result: 318.2870079121582
MAE_test=np.mean(np.abs((np.transpose(btc_model.predict(LSTM_test_inputs))+1)-
            (test_set['Close'].values[lag:])/(test_set['Close'].values[:-lag])))
print(MAE_test)

############### Plot the actual close price of test_set and the predicted close price ##############

fig, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(model_data[model_data['dates']>= split_date]['dates'][lag:], test_set['Close'][lag:], label='Actual')
ax1.plot(model_data[model_data['dates']>= split_date]['dates'][lag:],
         ((np.transpose(btc_model.predict(LSTM_test_inputs))+1) * test_set['Close'].values[:-lag])[0], 
         label='Predicted')
ax1.annotate('MAE_test: %.4f'%MAE_test, xy=(0.55, 0.85),  xycoords='axes fraction')#The point (x, y) to annotate and the coordinate system is determined by xycoords.

ax1.legend() # even though we have already configured the label in ax1.plot, we must to give ax1.legend to show the legends in the plot.

plt.show()
#Some comments on the results:
#It seems that our model is not so bad at predicting the next-day close price with low MAE. 





###################### LSTM ######################

#Importation de la dataframe
dt = pd.read_csv("btc_aud_tout.csv", sep = ",")

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import LeakyReLU
from sklearn.metrics import mean_squared_error as mse
from sklearn import metrics


signal = dt["close"]#On essaie de prédire le prix de cloture

def mise_en_forme(raw_data, lag_max):
    #Retourne l'array des variables explicatives et celle des variables expliquées
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
model.add(LSTM(50, input_shape=(n_steps, n_features), activation='relu'))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(x_train, y_train, epochs=200, verbose=0)

#Prediction
pred_train = model.predict(x_train, verbose=0)

pred_train_graph = pred_train.reshape(len(pred_train))

#MSE de l'échantillon train
mse_train = math.sqrt(mse(y_train, pred_train_graph))

line_train = go.Figure()

line_train.add_trace(go.Scatter(x=list(range(len(x_train))), y=y_train, name='vraies',
                         line=dict(color='firebrick', width=1)))
line_train.add_trace(go.Scatter(x=list(range(len(x_train))), y=pred_train_graph, name = 'pred',
                         line=dict(color='royalblue', width=1)))
line_train.write_html("pred_train_close_taux.html", auto_open = True)

line_train_sub = make_subplots(rows=3, cols=1)

line_train_sub.append_trace(go.Scatter(x=list(range(len(x_train))), y=y_train, name='vraies',
                         line=dict(color='firebrick', width=1)), row=1, col=1)
line_train_sub.append_trace(go.Scatter(x=list(range(len(x_train))), y=pred_train_graph, name = 'pred',
                         line=dict(color='royalblue', width=1)), row=1, col=1)
line_train_sub.append_trace(go.Scatter(x=list(range(len(x_train))), y=y_train, name='vraies',
                         line=dict(color='firebrick', width=1)), row=2, col=1)
line_train_sub.append_trace(go.Scatter(x=list(range(len(x_train))), y=pred_train_graph, name = 'pred',
                         line=dict(color='royalblue', width=1)), row=3, col=1)
line_train_sub.write_html("pred_train_close_taux_sub.html", auto_open = True)


pred_test = model.predict(x_test, verbose=0)


pred_test_graph = pred_test.reshape(len(pred_test))

#MSE de l'échantillon test
mse_test = math.sqrt(mse(y_test, pred_test_graph))


line_test = go.Figure()

line_test.add_trace(go.Scatter(x=list(range(len(x_test))), y=y_test, name='vraies',
                         line=dict(color='firebrick', width=1)))
line_test.add_trace(go.Scatter(x=list(range(len(x_test))), y=pred_test_graph, name = 'pred',
                         line=dict(color='royalblue', width=1)))
line_test.write_html("pred_test_close_taux.html", auto_open = True)



line_test_sub = make_subplots(rows=3, cols=1)

line_test_sub.append_trace(go.Scatter(x=list(range(len(x_test))), y=y_test, name='vraies',
                         line=dict(color='firebrick', width=1)), row=1, col=1)
line_test_sub.append_trace(go.Scatter(x=list(range(len(x_test))), y=pred_test_graph, name = 'pred',
                         line=dict(color='royalblue', width=1)), row=1, col=1)
line_test_sub.append_trace(go.Scatter(x=list(range(len(x_test))), y=y_test, name='vraies',
                         line=dict(color='firebrick', width=1)), row=2, col=1)
line_test_sub.append_trace(go.Scatter(x=list(range(len(x_test))), y=pred_test_graph, name = 'pred',
                         line=dict(color='royalblue', width=1)), row=3, col=1)
line_test_sub.write_html("pred_test_close_taux_sub.html", auto_open = True)


######################################### IV- Prise de décision d'investissement  #########################################

###########################################################################################################################





    

