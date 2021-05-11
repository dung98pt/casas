#!/usr/bin/env python3
import csv, os, sys 
sys.path.append(os.getcwd())
# print(sys.path)
from datetime import datetime
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from sklearn.model_selection import train_test_split
from model.keras_model import get_model
import data
import argparse

p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
p.add_argument('--n', dest='dataset_name', action='store', default='cairo', help='input', required = False)
p.add_argument('--w', dest='winSize', action='store', default='', help='input', required = False)
args = p.parse_args()
seed = 7
units = 64
epochs = 100
args_model = "biLSTM"
datasetName = args.dataset_name
maxLength = int(args.winSize)
if __name__ == '__main__':
    X, Y, dictActivities, word_id = data.loadDataCase1(datasetName, maxLength, "train")
    print("Activities: ", dictActivities)
    cvaccuracy = []
    cvscores = []
    Y = label_encoder.fit_transform(Y)
    k = 0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed, stratify=Y)
    print('X_train shape:', X_train.shape)
    print('y_train shape:', Y_train.shape)
    if 'Ensemble' in args_model:
        X_train_input = [X_train, X_train]
        X_test_input = [X_test, X_test]
    else:
        X_train_input = X_train
        X_test_input = X_test
    no_activities = len(dictActivities)
    input_dim = len(word_id)
    model = get_model(args_model, input_dim, no_activities, maxLength)
    # model.load_weights("model/model_1-cairo-20210409-103533.h5")
    modelname = model.name
    # checkpoint callback
    currenttime = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    csv_logger = CSVLogger("logging/log_case_1/training/csv/"+model.name + '-' + datasetName + '-' + str(currenttime) + '.csv')
    model_checkpoint = ModelCheckpoint(
        "logging/log_case_1/training/model/"+model.name + '-' + datasetName + '-' + str(currenttime) + '.h5',
        monitor ="val_loss",
        save_best_only=True)
    # train the model
    print('Begin training ...')
    # compute_class_weight
    class_weight = {}
    num_class = len(dictActivities)
    num_sample = len(Y_train)
    for i in set(Y_train):
        class_weight[i] = num_sample/num_class/list(Y_train).count(i)
    print("class_weight: ", class_weight)
    # class_weight = compute_class_weight('balanced', np.unique(Y), Y)  # use as optional argument in the fit function
    model.fit(X_train_input, Y_train, validation_data = (X_test_input, Y_test), epochs=epochs, batch_size=128, verbose=1, class_weight = class_weight,
              callbacks=[csv_logger, model_checkpoint])