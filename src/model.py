from dt_classifier import DTClassifier
from sklearn.externals import joblib
import numpy as np
import os

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
        self.train()
        self.predict()

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
