import random
import numpy as np
from model import Model

class ExIt:
    def __init__(self, model =Model().load("../models/dtree_654750_gini_best.model")):
        self.model = model

    def get_best_move(self, chips: int) -> int:
        return self.model.predict(np.array([[chips]]))[0]

    # def store_play_state(self, chips, best_move, reward, terminal =False):
    #     chips_remaining = chips - best_move
    #     self.memory.append((chips, best_move, reward, chips_remaining, terminal))

    # def tune_model(self):
    #     batch = random.sample(self.memory, 1)
    #     for chips, best_move, reward, chips_remaining, terminal in batch:
    #         q_update = reward
    #         if not terminal:
    #             q_update = (reward + 0.95 * self.get_best_move(chips_remaining))
    #         X_train = np.array([[chips]])
    #         y_train = np.array([[best_move]])
    #         weights = np.array([q_update])


if __name__ == '__main__':
    model = Model()
    model.new_trained(datafile="../data/data_20k.csv", name="iter10")
    model.print_sample()
    model.as_graph_image()
    model.save()
    model.confusion_matrix_as_plot()
    # model = Model().load("../models/dtree_608405_gini_best.model")
    # model.print_sample()
    # import pdb; pdb.set_trace()
    # best_move = ExIt().get_best_move(15)
