import numpy as np 
import pickle
import os

def loadDataCase1(datasetName, winSize=2000, use="train"):
    X = np.load('datasets/preprocess_cate1/{}_{}_X_{}.npy'.format(datasetName, winSize, use), allow_pickle=True)
    Y = np.load('datasets/preprocess_cate1/{}_{}_Y_{}.npy'.format(datasetName, winSize, use), allow_pickle=True)
    pickle_in = open("./datasets/activities_dictionary/{}_activity_list.pickle".format(datasetName),"rb")
    dictActivities = pickle.load(pickle_in)
    print(X.shape, len(Y), dictActivities)
    return X, Y, dictActivities

seed = 7
test_size = 0.3
np.random.seed(seed)

def loadDataCase2(datasetName, winSize=100, use="train"):
    pickle_in = open("./datasets/activities_dictionary/{}_activity_list.pickle".format(datasetName),"rb")
    dictActivities = pickle.load(pickle_in)
    *listActivities, = dictActivities
    X = np.load('datasets/preprocess_cate2/{}_{}_X_{}.npy'.format(datasetName, winSize, use), allow_pickle=True)
    Y = np.load('datasets/preprocess_cate2/{}_{}_Y_{}.npy'.format(datasetName, winSize, use), allow_pickle=True)
    # tokenize Y
    Y1 = Y
    for i, y in enumerate(Y):
        Y1[i]=dictActivities[y]
    Y = np.asarray(Y1)
    return X, Y, dictActivities, listActivities

def get_vocab_size(datasetName):
    if "milan"==datasetName:
        root_logdir = os.path.join("results", "logs_milan_sliding_windows_over_activity")
        vocabSize = 130
    if "aruba"==datasetName:
        root_logdir = os.path.join("results", "logs_aruba_sliding_windows_over_activity")
        vocabSize = 309
    if "cairo"==datasetName:
        root_logdir = os.path.join("results", "logs_cairo_sliding_windows_over_activity")
        vocabSize = 256
    return root_logdir, vocabSize