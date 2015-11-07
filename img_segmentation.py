#coding: utf-8

import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger

class ImageSegmentation:
    def __init__(self, class_weights, inference_method='qpbo', symmetric_edge_features=[0, 1], antisymmetric_edge_features=[2]):
        self.cw = class_weights
        self.im = inference_method
        self.sef = symmetric_edge_features
        self.aef = antisymmetric_edge_features

        print('model parameters:------')
        print self.cw
        print self.im
        print self.sef
        print self.aef
        print('-----------------------')
        self.model = crfs.EdgeFeatureGraphCRF(self.im,
                                 class_weight=self.cw,
                                 symmetric_edge_features=self.sef,
                                 antisymmetric_edge_features=self.aef)

def load_data(path):
    data_train = pickle.load(open(path))
    return data_train

def train(data_path, C=0.01, n_states=21, train_iter=100000):
    data_train = load_data(data_path)
    print("number of samples %s" % len(data_train['X']))

    class_weights = 1. / np.bincount(np.hstack(data_train['Y']))
    class_weights *= 21. / np.sum(class_weights)

    print(class_weights)

    iseg = ImageSegmentation(class_weights)
    
    experiment_name = "edge_features_one_slack_train_val_%f" % C

    ssvm = learners.NSlackSSVM(iseg.model, verbose=2, C=C, max_iter=train_iter, n_jobs=-1, tol=0.0001, show_loss_every=5, logger=SaveLogger(experiment_name + ".pickle", save_every=100), 
                                inactive_threshold=1e-3, inactive_window=10, batch_size=100)

    ssvm.fit(data_train['X'], data_train['Y'])

    return ""

def test():
    data_train = load_data("data/data_train.pickle")

    print("test: number of samples %s" % len(data_train['X']))
    print("test: number of targets %s" % len(data_train['Y']))

    print data_train['X'][0]
    #print data_train['Y'][0]
    
    class_weights = 1. / np.bincount(np.hstack(data_train['Y']))
    class_weights *= 21. / np.sum(class_weights)

    i = ImageSegmentation(class_weights)

if __name__ == "__main__":
    test()
    #train("data/data_train.pickle")
