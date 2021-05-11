import os 
import numpy as np
from data import loadDataCase1, loadDataCase2
from sklearn.utils import compute_class_weight
from evaluate.keras_evaluate import evaluate_model
# os.system("python pre_process.py --w1 2000 --w2 100")
os.system("python pre_process.py --n milan --t False --w1 2000")
os.system("python train\keras_train.py --n milan --w 2000")
# X, Y, listActivities = loadDataCase1("cairo", 2000, "train")
# print(listActivities, len(listActivities), set(Y), len(set(Y)))
# import tensorflow as tf
# sequence = [[1], [2, 3], [4, 5, 6]]
# print(tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=2))
# evaluate_model("evaluate/biLSTM-cairo-20210505-032112.h5")


