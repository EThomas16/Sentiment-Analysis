# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:44:28 2018

@author: Erik
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors

#accuracies = []

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
#modifies missing data, -99999 is treated as an outlier rather than missing data
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier(n_jobs=-1)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
examples_measures = example_measures.reshape(len(example_measures), -1)
#Predictions done for all arrays inside example_measures
prediction = clf.predict(example_measures)
"""
#print("Accuracy: {} Prediction: {}".format(accuracy, prediction))
accuracies.append(accuracy)
    
print(sum(accuracies) / len(accuracies))"""