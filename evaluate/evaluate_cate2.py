#!/usr/bin/env python3
import os, sys
sys.path.append(os.getcwd())
import numpy as np
import time
import csv
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from data import loadDataCase
from tensorflow.keras.utils import *
import argparse
p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
p.add_argument('--m', dest='best_model_path', action='store', default='', help='input', required = True)
p.add_argument('--n', dest='dataset_name', action='store', default='cairo', help='input', required = True)
p.add_argument('--c', dest='category', action='store', default='test', help='input', required = True)
p.add_argument('--w', dest='winSize', action='store', default='', help='input', required = False)
args = p.parse_args()

def evaluate_model(model, testX, testy, batch_size):
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

datasetName = args.dataset_name
winSize = args.winSize
best_model_path = args.best_model_path
saved_model = tf.keras.models.load_model(best_model_path)
saved_model.summary()
print(os.path.basename(best_model_path))
# tên vớ vẩn
model_name = saved_model.name
print(model_name)

X_TEST, Y_TEST, transition_mask, dictActivities, listActivities, vocab_size= loadDataCase(datasetName, winSize, args.category, True)
filename = "{}_{}".format(datasetName, winSize)
currenttime  = time.strftime("%Y_%m_%d_%H_%M_%S")
path = os.path.join("logging/log_case_2/results", model_name, "run_"+ filename + "_" + str(currenttime))
###########_FCN_##########
if model_name == "FCN":
    path_FCN = path
###########_FCN_WITH_EMBEDDING_##########
if model_name == "FCN_Embedded":
    path_FCN_Embedded = path
###########_LSTM_##########
if model_name == "LSTM":
    path_LSTM = path
###########_LSTM_WITH_EMBEDDING_##########
if model_name == "LSTM_Embedded":
    path_LSTM_Embedded = path
# create a folder with the model name
# if the folder doesn't exist
if not os.path.exists(path):
    os.makedirs(path)
# all paths
root_logdir = os.path.join("results", "logs")
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
##########_EVALUATION_##########
# load the best model on this k fold
cvscores = []
bscores = []

# evaluate
batch = 1024
x_test = X_TEST
y_test = to_categorical(Y_TEST, num_classes=max(Y_TEST)+1)
score = evaluate_model(saved_model, x_test, y_test, batch)
# store score
cvscores.append(score)
print('Accuracy: %.3f' % (score * 100.0))

# transition score
transition_x = np.concatenate([x_test[i].reshape(1, x_test[i].shape[0]) for i in transition_mask if i], axis=0)
transition_y = np.concatenate([y_test[i].reshape(1, y_test[i].shape[0]) for i in transition_mask if i], axis=0)
transition_score = evaluate_model(saved_model, transition_x, transition_y, batch)
print("transition_x: {}, transition_y: {}".format(transition_x.shape, transition_y.shape))
# store score
# cvscores.append(score)
print('Transition Accuracy: %.3f' % (transition_score * 100.0))
##########_GENERATE_##########
# Make prediction using the model
Y_hat = saved_model.predict(x_test)
Y_pred = np.argmax(Y_hat, axis=1)
Y_pred = Y_pred.reshape(Y_pred.shape[0], 1)
Y_pred = Y_pred.astype('int32')
Y_test = Y_TEST.astype('int32')
Y_test = Y_test.reshape(Y_test.shape[0], 1)
report = classification_report(Y_test, Y_pred, target_names=listActivities)
print(report)
text_file = open(report_path, "w")
n = text_file.write(report)
text_file.close()
cm=confusion_matrix(Y_test, Y_pred)
print(cm)
text_file = open(confusion_path, "w")
n = text_file.write("{}".format(cm))
text_file.close()
bscore = balanced_accuracy_score(Y_test, Y_pred)
bscores.append(bscore)
print('Balanced Accuracy: %.3f' % (bscore * 100.0))
print('Model: {}'.format(model_name))
print('Accuracy: {:.2f}% (+/- {:.2f}%)'.format(np.mean(cvscores)*100, np.std(cvscores)))
print('Balanced Accuracy: {:.2f}% (+/- {:.2f}%)'.format(np.mean(bscores)*100, np.std(bscores)))
print('Transition Accuracy: %.2f' % (transition_score * 100.0))

path = "logging/log_case_2/"
# save metrics
csvfile = 'cv_scores_' + model_name + '_' + filename + '_' + str(currenttime) + '.csv'
with open(os.path.join(path, csvfile), "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(["accuracy score :"])
    for val in cvscores:
        writer.writerow([val*100])
    writer.writerow([""])
    writer.writerow([np.mean(cvscores)*100])
    writer.writerow([np.std(cvscores)])
    writer.writerow([""])
    writer.writerow(["balanced accuracy score :"])
    for val2 in bscores:
        writer.writerow([val2*100])
    writer.writerow([""])
    writer.writerow([np.mean(bscores)*100])
    writer.writerow([np.std(bscores)])