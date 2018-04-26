import datetime
import numpy as np
import pandas as pd
from sklearn import naive_bayes, svm, metrics
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from matplotlib import style

#Uses a given style from matplotlib's style library
style.use('ggplot')

class SentimentAnalysis():
    def __init__(self, train_csv, test_csv, _class):
        self.df_train = pd.read_csv(train_csv)
        self.df_test = pd.read_csv(test_csv)
        #Acquires the features of the dataset by removing the review_score class
        self.X_train = np.array(self.df_train.drop([_class], 1))
        #Acquires the class of the dataset (review_score)
        self.y_train = np.array(self.df_train[_class])
        #Performs the same operation as was done to the training dataset
        self.X_test = np.array(self.df_test.drop([_class], 1))
        self.y_test = np.array(self.df_test[_class])
        
    def classify(self, clf):
        """"Fits the given classifier to the training data and predicts the test data"""
        #Used to record the time, start and end are placed before and after fitting of the data and predicting of the class
        start = datetime.datetime.now()
        clf.fit(self.X_train, self.y_train)
        predicted = clf.predict(self.X_test)
        end = datetime.datetime.now()
        total_time = end - start
        #Conversion required to get total seconds from datetime object
        total_time = total_time.total_seconds()
        #Calls the internal performance_score method to get the f_measure of the algorithm
        f_measure = self.performance_score(predicted)
        #Total time printed for visibility purposes
        print("Total time for algorithm completion {}".format(total_time))
        #Values used for graphing returned
        return f_measure, total_time
    
    def scale_classify(self, clf, lower_range=0, upper_range=1):
        """Normalises the features in a given range before fitting and classification"""
        #MinMaxScaler class takes an integer range for which to scale the features
        scaler = MinMaxScaler(feature_range=(lower_range, upper_range))
        #The training features are then refitted within the range using the scaler
        rescaledX = scaler.fit_transform(self.X_train)
        rescaledX_test = scaler.fit_transform(self.X_test)
        start = datetime.datetime.now()
        clf.fit(rescaledX, self.y_train)
        predicted = clf.predict(rescaledX_test)
        end = datetime.datetime.now()
        total_time = end - start
        total_time = total_time.total_seconds()
        f_measure = self.performance_score(predicted)
        print("Total time for algorithm completion {}".format(total_time))
        return f_measure, total_time
    
    def performance_score(self, predicted):
        """Internal method - gets the relevant information from the classification report"""
        #Score stores the entire classification report as a variable
        score = metrics.classification_report(self.y_test, predicted, digits=3)
        #With the report being printed for visibility
        print(score)
        #The relevant metrics are taken from the report. Precision and recall were not used for graphing but can be returned if required
        precision = score[178:184]
        recall = score[187:193]
        f_measure = score[198:203]
        return f_measure
        
    def graph_plot(self, x_vals, y_vals, x_label, y_label, label0='', label1=''):
        """Uses matplotlib's pyplot to plot a line graph with given data"""
        #Plots the given data on the axes at the given place. 
        #Label is a parameter that is used for creating a legend for the graph
        plt.plot(x_vals, y_vals, linewidth=2.0, label=label0)
        #Gives labels to each axis
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #For denoting a specific axis range and for showing the legend. Can be commented out if required
        plt.axis([1,10,0.0,1.0])
        plt.legend()
        #Shows the final graph
        plt.show()

#Each class initialised uses a different dataset. The first two parameters are the training and test csv files
#The last parameter is the title of the class of the dataset     
sent_a = SentimentAnalysis("Review_Dataset/reviews_Video_Games_training.csv", "Review_Dataset/reviews_Video_Games_test.csv", "review_score")
sent_b = SentimentAnalysis("Review_Dataset/lexicon_dataset_train.csv", "Review_Dataset/lexicon_dataset_test.csv", "review_score")
#Setting of variables of classifiers used for testing
clf_gnb = naive_bayes.GaussianNB()
clf_mnb = naive_bayes.MultinomialNB()
clf_lsvm = svm.SVC(kernel='linear')
clf_rbfsvm = svm.SVC(kernel='rbf')
clf_polysvm = svm.SVC(kernel='poly')
clf_sigmoidsvm = svm.SVC(kernel='sigmoid')
clf_liblinear = svm.LinearSVC()
