#!/usr/bin/env python3
import csv, os, sys 
sys.path.append(os.getcwd())
print(sys.path)
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

seed = 7
units = 64
epochs = 1
args_model = "biLSTM"
datasetName = "cairo"
maxLength = 2000
if __name__ == '__main__':
    X, Y, dictActivities = data.loadDataCase1(datasetName, maxLength, "train")
    print("Activities: ", dictActivities)
    cvaccuracy = []
    cvscores = []
    Y = label_encoder.fit_transform(Y)
    k = 0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
    print('X_train shape:', X_train.shape)
    print('y_train shape:', Y_train.shape)
    if 'Ensemble' in args_model:
        input_dim = len(X_train)
        X_train_input = [X_train, X_train]
        X_test_input = [X_test, X_test]
    else:
        input_dim = len(X_train)
        X_train_input = X_train
        X_test_input = X_test
    no_activities = len(dictActivities)
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
    class_weight = compute_class_weight('balanced', np.unique(Y),Y)  # use as optional argument in the fit function
    # model.fit(X_train_input, Y_train, validation_split=0.2, epochs=1, batch_size=64, verbose=1)
    model.fit(X_train_input, Y_train, validation_split=0.2, epochs=epochs, batch_size=64, verbose=1,
              callbacks=[csv_logger, model_checkpoint])