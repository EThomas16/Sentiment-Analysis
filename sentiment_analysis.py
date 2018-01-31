# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:45:11 2018

@author: Erik
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors, svm
from multiprocessing import Process

def csv_read():
    """Reads given .csv files and separates the data into features and classes"""
    #Stores the information from the training and testing .csv files into their respective dataframes
    df_train = pd.read_csv("Review_Dataset/reviews_Video_Games_training.csv")
    df_test = pd.read_csv("Review_Dataset/reviews_Video_Games_test.csv")
    
    #Acquires the features of the dataset by removing the review_score class
    X_train = np.array(df_train.drop(['review_score'], 1))
    #Acquires the class of the dataset (review_score)
    y_train = np.array(df_train['review_score'])
    #Performs the same operation as was done to the training dataset
    X_test = np.array(df_test.drop(['review_score'], 1))
    y_test = np.array(df_test['review_score'])
    
    return X_train, y_train, X_test, y_test
    
def knn_classify(X_train, y_train, X_test, y_test):
    """Classifies the given dataset using K-nearest-neighbours, then predicts instances and determines the accuracy of the classifier"""
    #Stores the KNN classifier
    #Changing n_jobs alters threading, n_neighbors alters accuracy (depending on dataset size and weighting)
    #For more information go to: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    clf = neighbors.KNeighborsClassifier(n_jobs=-1, n_neighbors=15)
    #Fits the training data to the classifier
    clf.fit(X_train, y_train)
    #Handles the writing of predictions to a given .txt file
    f_name = "knn_predict.txt"
    predictions = clf.predict(X_test)
    predict_write(f_name, predictions)
    #Accuracy is determined from using the classifier on the testing data
    accuracy = clf.score(X_test, y_test)
    print("The accuracy of KNN is {}".format(accuracy))

def svm_classify(X_train, y_train, X_test, y_test):
    """Classifies the given dataset using a support vector machine, then predicts instances and determines the accuracy of the classifier"""
    #Stores the SVM classifier
    #For parameters to alter go to: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    #The code below has the same functionality as in knn_classify, except it performs it on the svm classifier
    f_name = "svm_predict.txt"
    predictions = clf.predict(X_test)
    predict_write(f_name, predictions)
    
    accuracy = clf.score(X_test, y_test)
    print("The accuracy of SVM is {}".format(accuracy))
    
def predict_write(f_name, predictions):
    """Used to write predictions for an algorithm to a specified file (f_name)"""
    f_name = "svm_predict.txt"
    #Empties the currently selected file
    with open(f_name, "w"): pass
    #Used to state the instance number in the writelines statement
    counter = 1
    #Loops through the predictions array and writes each instance's prediction to a specified file (f_name)
    for predicted_label in predictions:
        #Uses a+ to append to the file rather than to overwrite each line
        predict_file = open(f_name, "a+")
        predict_file.writelines("The predicted label of instance {} is {}\n".format(counter, predicted_label))
        #Increments the counter by one for the next instance to be predicted
        counter += 1
    #File must be closed to prevent issues with IO
    predict_file.close()


if __name__ == '__main__':
    #Reads the training and testing data csvs to get the relevant feature and class data
    X_train, y_train, X_test, y_test = csv_read()
    #Runs the two algorithms in parallel to increase compute speed
    proc_knn = Process(target=knn_classify, args=(X_train, y_train, X_test, y_test))
    proc_knn.start()
    proc_svm = Process(target=svm_classify, args=(X_train, y_train, X_test, y_test))
    proc_svm.start()