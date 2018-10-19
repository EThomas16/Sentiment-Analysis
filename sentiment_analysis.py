import sys
import os

import numpy as np
import pandas as pd
from sklearn import naive_bayes, svm, metrics
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from matplotlib import style

from decorators import timer

# used to style any graphs created using matplotlib
style.use('ggplot')

class SentimentData():
    def __init__(self, train_path: str, test_path: str, lbl: str):
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        self.X_train = np.array(df_train.drop([lbl], 1))
        self.y_train = np.array(df_train[lbl])
        self.X_test = np.array(df_test.drop([lbl], 1))
        self.y_test = np.array(df_test[lbl])

    @staticmethod
    def scan_data_dir(data_dir: str, ext: str = '.csv') -> list:
        """
        Scans a given directory to find all dataset files (isolates them from other file types)

        Keyword arguments:
        data_dir -- the directory containing the data in question
        ext -- the file extension of the files to be extracted

        Returns:
        data_files -- a list of all files of a given extension in the data_dir directory
        """
        data_files = []

        for path, subdir, files in os.walk(data_dir):
            for f_name in files:
                if ext in f_name:
                    f_path = os.path.join(path, f_name)
                    data_files.append(f_path)

        return data_files

    @staticmethod
    def setup_data(data_files: list, data_name: str) -> (str, str):
        """
        Finds the training and test data within the files of the given directory

        Keyword arguments:
        data_files -- a list of all dataset files in a given directory
        data_name -- the keyword(s) used to identify the dataset being read

        Returns:
        train_path -- the full path to the training data, to be used in classification
        test_path -- the full path to the testing data, to be used in classification
        """
        data_files = SentimentData.scan_data_dir(rel_path)
        for f_data in data_files:
            if data_name in f_data:
                if "train" in f_data:   train_path = f_data
                elif "test" in f_data:  test_path = f_data

        return train_path, test_path

@timer  
def classify(clf, sent_data: object) -> float:
    """"
    Fits the given classifier to the training data and predicts the test data

    Keyword arguments:
    clf -- the classifier to be used to predict the data
    sent_data -- the instance of the SentimentData class containing the correct dataset to be tested

    Returns:
    score_list -- a list of all of the metrics for the classifier of the given dataset
    """
    clf.fit(sent_data.X_train, sent_data.y_train)
    predicted = clf.predict(sent_data.X_test)
    score_list = performance_score(predicted, sent_data)

    return score_list

@timer
def scale_clf(clf, sent_data: object, lower_range: int = 0, upper_range: int = 1) -> list:
    """
    Normalises the features in a given range before fitting and classification

    Keyword arguments:
    clf -- the classifier to be used to predict the data
    sent_data -- the instance of the SentimentData class containing the correct dataset to be tested
    lower_range -- the lower value of normalisation for the dataset's values
    upper_range -- the higher value of normalisation for the dataset's values

    Returns:
    score_list -- a list of all of the metrics for the classifier of the given dataset
    """
    # MinMaxScaler class takes an integer range for which to scale the features
    scaler = MinMaxScaler(feature_range=(lower_range, upper_range))
    # normalises the data using the instance of the scaler class 
    # does this between the lower and upper range
    rescaledX = scaler.fit_transform(sent_data.X_train)
    rescaledX_test = scaler.fit_transform(sent_data.X_test)

    clf.fit(rescaledX, sent_data.y_train)
    predicted = clf.predict(rescaledX_test)
    score_list = performance_score(predicted, sent_data)

    return score_list

def performance_score(predicted, sent_data: object, all_metrics: bool = False) -> list:
    """
    Gets the relevant information from scikit-learn's classification report

    Keyword arguments:
    predicted -- the predicted data from a given classifier, required to create a classification report
    sent_data -- the instance of the SentimentData class containing the correct dataset to be tested
    all_metrics -- a boolean check for the user to decide whether they also return precision and recall alongside f-measure

    Returns:
    score_list -- a list of all of the metrics for the classifier of the given dataset
    """
    score = metrics.classification_report(sent_data.y_test, predicted, digits=3)
    # printed for visibility
    print(score)
    # a list is used to give the user the option of utilising all three metrics or just one
    score_list = []

    if all_metrics:
        precision = score[178:184]
        recall = score[187:193]
        score_list = [precision, recall]

    f_measure = score[198:203]
    score_list.append(f_measure)

    return score_list

if __name__ == "__main__":
    # each class initialised uses a different dataset. The first two parameters are the training and test csv files
    lbl = "review_score"   
    rel_path = "Review_Dataset/"
    ext = ".csv"

    data_files = SentimentData.scan_data_dir(rel_path)

    train_path, test_path = SentimentData.setup_data(data_files, data_name="reviews_Video_Games")
    sent_games = SentimentData(train_path, test_path, lbl)

    train_path, test_path = SentimentData.setup_data(data_files, data_name="lexicon")
    sent_lex = SentimentData(train_path, test_path, lbl)

    clf_gnb = naive_bayes.GaussianNB()
    clf_mnb = naive_bayes.MultinomialNB()
    clf_lsvm = svm.SVC(kernel='linear')
    clf_liblinear = svm.LinearSVC()

    score_list = classify(clf_liblinear, sent_games)