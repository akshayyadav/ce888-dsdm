import numpy as np
from random import randint
from model import Model

class ExIt:
    def __init__(self, model =Model().load("../models/dtree_608405_gini_best.model")):
        self.model = model

    def get_best_move(self, chips: int) -> int:
        return self.model.predict(np.array([[chips]]))[0]


if __name__ == '__main__':
    model = Model()
    model.new_trained(datafile="../data/data_20k.csv")
    model.print_sample()
    model.as_graph_image()
    model.save()
# model = Model().load("../models/dtree_487610_gini_best.model")
# model.print_sample()
# predicted = model.predict(np.array([[5,2,1]]))
# print(predicted)
# import pdb; pdb.set_trace()
# best_move = ExIt().get_best_move(15)
