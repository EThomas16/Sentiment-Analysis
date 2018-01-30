# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 14:59:42 2018

@author: Erik
"""
import numpy as np
import warnings
import pandas as pd
import random
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib import style
from collections import Counter

#style.use('fivethirtyeight')
#Manual calculation
"""plot_1 = [1,3]
plot_2 = [2,5]
#two dimensions so i is 2
euclidean_distance = sqrt((plot_1[0] - plot_2[0])**2 + (plot_1[1] - plot_2[1])**2)
#Two classes and features
dataset = {'k': [[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

for i in dataset:
    for i_ in dataset[i]:
        plt.scatter(i_[0], i_[1], s=100, color=i)

plt.scatter(new_features[0], new_features[1], s=100)

plt.show()"""

def knn(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')
    
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
            
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence

accuracies = []
for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)
    
    test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    #First 20% of data
    train_data = full_data[:-int(test_size*len(full_data))]
    #Last 20% of data
    test_data = full_data[-int(test_size*len(full_data)):]
    
    for i in train_data:
        #-1 takes last element
        train_set[i[-1]].append(i[:-1])
        
    for i in test_data:
        #-1 takes last element
        test_set[i[-1]].append(i[:-1])
        
    correct = 0
    total = 0
    
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = knn(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
            accuracies.append(correct/total)
            
print(sum(accuracies) / len(accuracies))
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        