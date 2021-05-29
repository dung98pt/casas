#!/usr/bin/env python3
import os, sys
sys.path.append(os.getcwd())
import io
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt
import argparse
import csv
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.activations import *
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from model.tf_model import FCN, FCNEmbedded, LSTM, LSTMEmbedded
from data import loadDataCase

import argparse
p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
p.add_argument('--n', dest='dataset_name', action='store', default='cairo', help='input', required = False)
p.add_argument('--w', dest='winSize', action='store', default='', help='input', required = False)
args = p.parse_args()

seed = 7
epoch = 200
batch = 1024
verbose = True
patience = 30
np.random.seed(seed)
datasetName = args.dataset_name
winSize = args.winSize

def load_data(datasetName, winSize):
    X_TRAIN, Y_TRAIN, _, listActivities, vocabSize = loadDataCase(datasetName, winSize, "train", "case2")
    X_TEST, Y_TEST, _, listActivities, _ = loadDataCase(datasetName, winSize, "test", "case2")
    X_TRAIN, X_VALIDATION, Y_TRAIN, Y_VALIDATION = train_test_split(X_TRAIN, Y_TRAIN, test_size=0.2, random_state=seed, stratify=Y_TRAIN)
    return X_TRAIN, Y_TRAIN, X_VALIDATION, Y_VALIDATION, X_TEST, Y_TEST, listActivities, vocabSize

def evaluate_model(model, testX, testy, batch_size):
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

# serialize model to JSON
def save_model(model,filename):
    model_json = model.to_json()
    with open(filename+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(filename+".h5")
    print("Saved model to disk")

# load json and create model
def load_model_2(filename):
    json_file = open(filename+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(filename+".h5")
    return loaded_model

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='None', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)
  # Normalize the confusion matrix.
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 1.05
  #threshold = 10
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if cm[i, j] > threshold:
        color = "white"  
    else: 
        color = "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

if __name__ == '__main__':
    filename = "{}_{}".format(datasetName, winSize)
    # MODELS = ['LSTM_Embedded','LSTM','FCN','FCN_Embedded']
    root_logdir = os.path.join("results", "logs_sliding_windows_over_activity")
    print(tf.__version__)
    strategy = tf.distribute.MirroredStrategy()
    X_TRAIN, Y_TRAIN, X_VALIDATION, Y_VALIDATION, X_TEST, Y_TEST, listActivities, vocabSize = load_data(datasetName, winSize)
    # compute_class_weight
    class_weight = {}
    num_class = len(listActivities)
    num_sample = len(Y_TRAIN)
    num_classes = max(Y_TRAIN) + 1
    print("num_classes: ", num_classes)
    for i in set(Y_TRAIN):
        class_weight[i] = num_sample/num_class/list(Y_TRAIN).count(i)
    print("class_weight: ", class_weight)
    # attention
    cvscores = []
    bscores = []
    path_attention = ""
    currenttime  = time.strftime("%Y_%m_%d_%H_%M_%S")

    # for k in range(len(X_TRAIN)):
    y_train = to_categorical(Y_TRAIN, num_classes=num_classes)
    y_validation = to_categorical(Y_VALIDATION, num_classes=num_classes)
    y_test = to_categorical(Y_TEST, num_classes=num_classes)
    #region prepare model
    ###########_FCN_##########
    x_train = X_TRAIN
    x_validation = X_VALIDATION
    x_test = X_TEST
    # model = attention.model(x_train, y_train, vocabSize)
    use_model = "FCNEmbedded"
    if use_model == "FCNEmbedded":
        model = FCNEmbedded.modelFCNEmbedded(x_train, y_train, vocabSize)
    model_name = model.name
    path = os.path.join("logging/log_case_2/results", model_name, "run_"+ filename + "_" + str(currenttime))
    if not os.path.exists(path):
        os.makedirs(path)
    # all paths
    run_id = model_name + "_" + filename + "_" + str(currenttime)
    log_dir = os.path.join(root_logdir, run_id)
    csv_name = model_name + "_" + filename + "_" + ".csv"
    csv_path = os.path.join(path, csv_name)
    picture_name = model_name + "_" + filename + "_" + ".png"
    picture_path = os.path.join(path, picture_name)
    report_name = model_name + "_repport_" + filename + "_" + ".txt"
    report_path = os.path.join(path, report_name)
    confusion_name = model_name + "_confusion_matrix_" + filename + "_" + ".txt"
    confusion_path = os.path.join(path, confusion_name)
    model_name_saved = model_name + "_" + filename
    model_path = os.path.join(path, model_name_saved)
    best_model_name_saved = model_name + "_" + filename + "_BEST_" +".h5"
    best_model_path = os.path.join(path, best_model_name_saved)
    #endregion
    #ceate a picture of the model
    print(picture_path)
    # plot_model(model, show_shapes=True, to_file=picture_path)
    #compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print summary
    print(model.summary())
    # create a folder with the log
    # if the folder doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # create a callback for the tensorboard
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir)
    #file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
    #callbacks
    csv_logger = CSVLogger(csv_path)
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    mc = ModelCheckpoint(best_model_path, monitor="val_loss", mode='min', verbose=1, save_best_only=True)
    # Define the per-epoch callback.

    #cbs = [csv_logger,tensorboard_cb,mc,es,cm_callback]
    cbs = [csv_logger,tensorboard_cb,mc,es]
    # fit network
    model.fit(x_train, y_train, epochs=50, batch_size=batch, verbose=verbose, callbacks=[csv_logger,tensorboard_cb,mc], validation_data=(x_validation, y_validation), class_weight = class_weight)
    model.fit(x_train, y_train, epochs=epoch-50, batch_size=batch, verbose=verbose, callbacks=cbs, validation_data=(x_validation, y_validation), class_weight = class_weight)
    os.system("python evaluate/evaluate_cate2.py --m {} --n {} --w {}".format(best_model_path, datasetName, winSize))
