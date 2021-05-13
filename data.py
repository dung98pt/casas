import numpy as np 
from utilities import load_dict, label_encode
import os

def loadDataCase1(datasetName, winSize=2000, use="train"):
    X = np.load('datasets/preprocess_cate1/{}_{}_X_{}.npy'.format(datasetName, winSize, use), allow_pickle=True)
    Y = np.load('datasets/preprocess_cate1/{}_{}_Y_{}.npy'.format(datasetName, winSize, use), allow_pickle=True)
    activities_dict_path = "./datasets/activities_dictionary/{}_activity_list.pickle".format(datasetName)
    word_id_path = "./datasets/word_id/{}.pickle".format(datasetName)
    word_id = load_dict(word_id_path)
    dictActivities = load_dict(activities_dict_path)
    print(X.shape, len(Y), len(dictActivities), len(word_id))
    return X, Y, dictActivities, word_id

def loadDataCase(datasetName, winSize=2000, use="train", _type="case1"):
    if _type == "case1":
        folder="preprocess_cate1"
    elif _type == "case2":
        folder="preprocess_cate2"
    X = np.load('datasets/{}/{}_{}_X_{}.npy'.format(folder, datasetName, winSize, use), allow_pickle=True)
    Y = np.load('datasets/{}/{}_{}_Y_{}.npy'.format(folder, datasetName, winSize, use), allow_pickle=True)
    activities_dict_path = "./datasets/activities_dictionary/{}_activity_list.pickle".format(datasetName)
    word_id_path = "./datasets/word_id/{}.pickle".format(datasetName)
    word_id = load_dict(word_id_path)
    vocab_size = len(word_id)
    dictActivities = load_dict(activities_dict_path)
    *listActivities, = dictActivities
    print(X.shape, len(Y), len(dictActivities), len(word_id), vocab_size)
    return X, Y, dictActivities, listActivities, vocab_size

seed = 7
test_size = 0.3
np.random.seed(seed)

def loadDataCase2(datasetName, winSize=100, use="train"):
    activities_dict_path = "./datasets/activities_dictionary/{}_activity_list.pickle".format(datasetName)
    dictActivities = load_dict(activities_dict_path)
    *listActivities, = dictActivities
    X = np.load('datasets/preprocess_cate2/{}_{}_X_{}.npy'.format(datasetName, winSize, use), allow_pickle=True)
    Y = np.load('datasets/preprocess_cate2/{}_{}_Y_{}.npy'.format(datasetName, winSize, use), allow_pickle=True)
    # print("ahihi", set(Y))
    # # tokenize Y
    # print("ahuhu", dictActivities)
    # Y1 = Y
    # for i, y in enumerate(Y):
    #     Y1[i]=dictActivities[y]
    # Y = np.asarray(Y1)
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