import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import make_scorer, classification_report, confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import time



class DTClassifier:
    def __init__(self, datafile ="../data/data_10k.csv", datafile_headers = ["chips", "lastMoved", "playerJustMoved", "playerNext", "winner"], criterion ="gini"):
        self.data_headers = datafile_headers
        self.data         = pd.read_csv(datafile, delimiter = ",", names = self.data_headers)
        self.clf          = DecisionTreeClassifier(criterion=criterion)
        self.features     = ["chips"]
        self.labels       = ["lastMoved"]
        self.trained      = False
        self.predicted    = False

    def split_train_test(self, test_portion =0.3, seed =1):
        X             = self.data[self.features].copy()
        y             = self.data[self.labels].copy()
        self.scores   = cross_val_score(self.clf, X, y, cv=10,scoring = make_scorer(acc))
        # 70% training and 30% test by default
        print(f'Spliting test/train with {test_portion*100}/{(1-test_portion)*100} split and seed = {seed}')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_portion, random_state=seed)

    def train(self):
        try: self.X_train, self.y_train
        except AttributeError: self.split_train_test()
        finally:
            # Train Decision Tree Classifer
            self.clf.fit(self.X_train,self.y_train)
            self.trained = True

    def predict(self, state =None):
        if self.trained:
            if state is None:
                print('No state given using test data')
                state = self.X_test
            # Predict the response for test dataset
            self.y_pred = self.clf.predict(state)
            self.predicted = True
            return self.y_pred
        else:
            raise TypeError("Train the classifier with 'train' method first!")

    def estimated_accuracy(self):
        try: self.scores
        except AttributeError: self.split_train_test()
        finally:
            # The mean score and the 95% confidence interval of the score estimate
            print("Estimated accuracy: %0.2f (+/- %0.2f)" % (self.scores.mean(), self.scores.std() * 2))

    def actual_prediction_accuracy(self):
        if self.predicted:
            print("Prediction accuracy:",acc(self.y_test, self.y_pred))
        else:
            raise TypeError("Predictions for model are not available, use 'predict' method first!")

    def confusion_matrix(self):
        if self.predicted:
            self.cfm = confusion_matrix(self.y_test, self.y_pred)
            acc = np.sum(self.cfm.diagonal()) / np.sum(self.cfm)
            print('Overall accuracy: {} %'.format(acc*100))
        else:
            raise TypeError("Predictions for model are not available, use 'predict' method first!")

    def classification_report(self):
        if self.predicted:
            print(classification_report(self.y_test, self.y_pred))
        else:
            raise TypeError("Predictions for model are not available, use 'predict' method first!")


    def print_sample(self):
        print("Data Sample:")
        print(self.data.head())
        self.estimated_accuracy()
        self.actual_prediction_accuracy()
        self.confusion_matrix()
        print("Confusion Matrix:")
        print(self.cfm)
        print("Classification Report:")
        self.classification_report()
        # print("Train data:")
        # print(f'X_train: {self.X_train.shape}')
        # print(f'X_train: {self.X_train.head}')
        # print(f'y_train: {self.y_train.head}')
        # print(f'X_test: {self.X_test.head}')
        # print(f'y_test: {self.y_test.head}')

    def as_graph_image(self):
        out_filename=f'dtree_{self.data.size}_{self.clf.criterion}_{self.clf.splitter}_{time.strftime("%Y%m%d_%H%M%S")}.png'
        out_filepath=f'{os.path.dirname(os.path.abspath(__file__))}/../images/{out_filename}'
        dot_data = StringIO()
        export_graphviz(self.clf, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True,feature_names = self.features, class_names=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(out_filepath)
        Image(graph.create_png())
        print(f'Generated node graph image: {out_filepath}')
