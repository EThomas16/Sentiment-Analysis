# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:05:50 2018

@author: Erik
"""

from statistics import mean
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import random

style.use('fivethirtyeight')
#test data
#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)
#true = positive correlation    false = negative correlation
def create_dataset(length, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(length):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
            
    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit(xs, ys):
    m = ((mean(xs) * mean(ys)) - mean(xs*ys)) / ((mean(xs)*mean(xs)) - mean(xs*xs))
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    #**2 denotes squaring a value
    return sum((ys_line-ys_orig)**2)

def coeff_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

xs, ys = create_dataset(40, 10, 2, correlation='pos')

m, b = best_fit(xs, ys)

regression_line = [(m*x) + b for x in xs]

predict_x = 60
predict_y = (m*predict_x) + b

r_squared = coeff_of_determination(ys, regression_line)
print("r-squared: {}".format(r_squared))
print("Gradient: {}, Intersect: {}".format(m, b))

plt.scatter(xs,ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()