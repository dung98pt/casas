import numpy as np 
from utilities import load_dict, label_encode
import os

def loadDataCase(datasetName, winSize, use="train"):
    X = np.load('datasets/processed_data/{}_{}_X_{}.npy'.format(datasetName, winSize, use), allow_pickle=True)
    Y = np.load('datasets/processed_data/{}_{}_Y_{}.npy'.format(datasetName, winSize, use), allow_pickle=True)
    activities_dict_path = "./datasets/activities_dictionary/{}_activity_list.pickle".format(datasetName)
    word_id_path = "./datasets/word_id/{}.pickle".format(datasetName)
    word_id = load_dict(word_id_path)
    vocab_size = len(word_id)
    dictActivities = load_dict(activities_dict_path)
    *listActivities, = dictActivities
    print(X.shape, len(Y), len(dictActivities), len(word_id), vocab_size)
    return X, Y, dictActivities, listActivities, vocab_size
