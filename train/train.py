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
from model.tf_model import FCN, FCNEmbedded, LSTM, LSTMEmbedded, attention
from data import loadDataCase

seed = 7
epoch = 2
batch = 256
verbose = True
patience = 20
np.random.seed(seed)
def load_data(datasetName, winSize):
    X_TRAIN, Y_TRAIN, _, listActivities, vocabSize = loadDataCase(datasetName, winSize, "train", "case1")
    X_TEST, Y_TEST, _, listActivities, _ = loadDataCase(datasetName, winSize, "test", "case1")
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
    datasetName = "cairo"
    winSize = 2000
    filename = "{}_{}".format(datasetName, winSize)
    # MODELS = ['LSTM_Embedded','LSTM','FCN','FCN_Embedded']
    root_logdir = os.path.join("results", "logs_sliding_windows_over_activity")
    print(tf.__version__)
    strategy = tf.distribute.MirroredStrategy()
    X_TRAIN, Y_TRAIN, X_VALIDATION, Y_VALIDATION, X_TEST, Y_TEST, listActivities, vocabSize = load_data(datasetName, winSize)
    # attention
    cvscores = []
    bscores = []
    path_attention = ""
    currenttime  = time.strftime("%Y_%m_%d_%H_%M_%S")

    # for k in range(len(X_TRAIN)):
    y_train = to_categorical(Y_TRAIN)
    y_validation = to_categorical(Y_VALIDATION)
    y_test = to_categorical(Y_TEST)
    #region prepare model
    ###########_FCN_##########
    x_train = X_TRAIN
    x_validation = X_VALIDATION
    x_test = X_TEST
    model = attention.model(x_train, y_train, vocabSize)
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
    mc = ModelCheckpoint(best_model_path, monitor="val_loss", mode='max', verbose=1, save_best_only=True)
    # Define the per-epoch callback.

    #cbs = [csv_logger,tensorboard_cb,mc,es,cm_callback]
    cbs = [csv_logger,tensorboard_cb,mc,es]
    # fit network
    model.fit(x_train, y_train, epochs=epoch, batch_size=batch, verbose=verbose, callbacks=cbs, validation_data=(x_validation, y_validation))
    # ##########_EVALUATION_##########
    # # load the best model on this k fold
    # saved_model = tf.keras.models.load_model(best_model_path)
    # # evaluate
    # score = evaluate_model(saved_model, x_test, y_test, batch)
    # # store score
    # cvscores.append(score)
    # print('Accuracy: %.3f' % (score * 100.0))
    # ##########_GENERATE_##########
    # # Make prediction using the model
    # Y_hat = saved_model.predict(x_test)
    # Y_pred = np.argmax(Y_hat, axis=1)
    # Y_pred = Y_pred.reshape(Y_pred.shape[0], 1)
    # Y_pred = Y_pred.astype('int32')
    # Y_test = Y_TEST.astype('int32')
    # report = classification_report(Y_test, Y_pred, target_names=listActivities)
    # print(report)
    # text_file = open(report_path, "w")
    # n = text_file.write(report)
    # text_file.close()
    # cm=confusion_matrix(Y_test, Y_pred)
    # print(cm)
    # text_file = open(confusion_path, "w")
    # n = text_file.write("{}".format(cm))
    # text_file.close()
    # bscore = balanced_accuracy_score(Y_test, Y_pred)
    # bscores.append(bscore)
    # print('Balanced Accuracy: %.3f' % (bscore * 100.0))

    # path = path_attention
    # print('Model: {}'.format(m))
    # print('Accuracy: {:.2f}% (+/- {:.2f}%)'.format(np.mean(cvscores)*100, np.std(cvscores)))
    # print('Balanced Accuracy: {:.2f}% (+/- {:.2f}%)'.format(np.mean(bscores)*100, np.std(bscores)))
    # # save metrics
    # csvfile = 'cv_scores_' + m + '_' + filename + '_' + str(currenttime) + '.csv'
    # with open(os.path.join(path, csvfile), "w") as output:
    #     writer = csv.writer(output, lineterminator='\n')
    #     writer.writerow(["accuracy score :"])
    #     for val in cvscores:
    #         writer.writerow([val*100])
    #     writer.writerow([""])
    #     writer.writerow([np.mean(cvscores)*100])
    #     writer.writerow([np.std(cvscores)])
    #     writer.writerow([""])
    #     writer.writerow(["balanced accuracy score :"])
    #     for val2 in bscores:
    #         writer.writerow([val2*100])
    #     writer.writerow([""])
    #     writer.writerow([np.mean(bscores)*100])
    #     writer.writerow([np.std(bscores)])
    
