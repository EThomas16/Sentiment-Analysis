# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:58:27 2018

@author: Erik
"""

import pandas as pd
import quandl, math, datetime
import numpy as np
#cross_validation for training and testing samples
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low',  'Adj. Close',  'Adj. Volume' ]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
#Predicts 10% out of dataframe, math.ceil rounds the value up
forecast_out = int(math.ceil(0.1*len(df)))
#Sets the label to be the forecasted value column
df['label'] = df[forecast_col].shift(-forecast_out)
#X stores features and takes everything that isn't a label
X = np.array(df.drop(['label'], 1))
#Normalisation of data size
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])
#Shuffles data keeping connections between x and y
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
#Sets up classifier using features and labels
#Can change this to use svm if required
#kernels can change the accuracy, explained more with SVMs
#clf = svm.SVR(kernel='poly')
#-1 for n_jobs runs as many processes as possible
#Does not need to be run every time, saved as classifier using pickle
"""clf = LinearRegression(n_jobs=10)
clf.fit(X_train, y_train)
with open('linearregression.pickle', 'wb') as file:
    pickle.dump(clf, file)"""

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
#Gives data to predict with
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
#no. of seconds in a day
one_day = 86400
next_unix = last_unix + one_day
#Fills the forecasted values
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()























