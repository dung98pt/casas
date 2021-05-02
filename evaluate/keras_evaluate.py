import sys, os
sys.path.append(os.getcwd())
from model.keras_model import get_model
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np 
import data
import csv
from datetime import datetime
from keras.utils.vis_utils import plot_model

def load_test_data(dataset_name):
  X_test = np.load("test_data/{}-x.npy".format(dataset_name), allow_pickle=True)
  Y_test = np.load("test_data/{}-y.npy".format(dataset_name), allow_pickle=True)
  input_dim, dictActivities = np.load("test_data/{}-dict.npy".format(dataset_name), allow_pickle=True)
  print(X_test.shape, Y_test.shape, input_dim, dictActivities)
  return X_test, Y_test, input_dim, dictActivities

def evaluate_model(model_path):
  cvaccuracy = []
  cvscores = []
  model_name = model_path.split("/")[-1]
  dataset = model_name.split("-")[1]
  model_structure_name = model_name.split("-")[0]
  X_test, Y_test, dictActivities, input_dim = load_test_data(dataset)
  if "Ensemble" in model_name:
    X_test_input = [X_test, X_test]
  else:
    X_test_input = X_test
  model = get_model(model_structure_name, input_dim, len(dictActivities))
  model.load_weights(model_path)
  plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  modelname = model.name
  target_names = sorted(dictActivities, key=dictActivities.get)
  print('Begin testing ...')
  # evaluate the model
  scores = model.evaluate(X_test_input, Y_test, batch_size=64, verbose=1)
  print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

  print('Report:')
  target_names = sorted(dictActivities, key=dictActivities.get)

  classes = model.predict_classes(X_test_input, batch_size=64)
  print(classification_report(list(Y_test), classes, target_names=target_names))
  print('Confusion matrix:')
  labels = list(dictActivities.values())
  cvaccuracy.append(scores[1] * 100)
  cvscores.append(scores)
  print(confusion_matrix(list(Y_test), classes, labels))
  print('{:.2f}% (+/- {:.2f}%)'.format(np.mean(cvaccuracy), np.std(cvaccuracy)))

  currenttime = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
  csvfile = 'score/cv-scores-' + modelname + '-' + dataset + '-' + str(currenttime) + '.csv'

  with open(csvfile, "w") as output:
      writer = csv.writer(output, lineterminator='\n')
      for val in cvscores:
          writer.writerow([",".join(str(el) for el in val)])
