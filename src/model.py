import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dt_classifier import DTClassifier
from sklearn.externals import joblib
from sklearn.utils.multiclass import unique_labels

class Model(DTClassifier):
    def load(self, filepath):
        return joblib.load(filepath)

    def save(self):
        filename=f'dtree_{self.data.size}_{self.clf.criterion}_{self.clf.splitter}.model'
        filepath=f'{os.path.dirname(os.path.abspath(__file__))}/../models/{filename}'
        joblib.dump(self, filepath)

    def new_trained(self, datafile, name ="default"):
        self.name = name
        DTClassifier.__init__(self,datafile=datafile)
        self.split_train_test()
        self.train()
        self.predict()

    def confusion_matrix_as_plot(self, filepath =None, title ="Confusion matrix", figsize =(10,10)):
        if filepath is None:
            filename=f'cfm_for_{self.name}_{self.data.size}_{self.clf.criterion}_{self.clf.splitter}.png'
            filepath=f'{os.path.dirname(os.path.abspath(__file__))}/../images/{filename}'
        labels=unique_labels(self.y_test, self.y_pred)
        cfm = self.cfm
        sum_cfm = np.sum(cfm, axis=1, keepdims=True)
        percent_cfm = cfm / sum_cfm.astype(float) * 100
        annot = np.empty_like(cfm).astype(str)
        rows, cols = cfm.shape
        for row in range(rows):
            for col in range(cols):
                annot[row, col] = '%.1f%%\n%d/%d' % (percent_cfm[row, col], cfm[row, col], sum_cfm[row])
        cfm_as_dataframe = pd.DataFrame(cfm, index=labels, columns=labels)
        cfm_as_dataframe.index.name = 'Actual'
        cfm_as_dataframe.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        sns.heatmap(cfm_as_dataframe, annot=annot, fmt='', ax=ax)
        plt.savefig(filepath)

if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    model = Model().load("../models/dtree_608405_gini_best.model")
    # model = Model().load("../models/dtree_491255_gini_best.model")
    # model = Model().load("../models/dtree_487610_gini_best.model")
    # model = Model()
    # model.new_trained(datafile="../data/data_20k.csv")
    model.print_sample()
    # model.save()
    predicted = model.predict(np.array([[15]]))
    print(predicted)
    model.confusion_matrix_as_plot()
