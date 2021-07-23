import os 
import numpy as np
from sklearn.utils import compute_class_weight

# os.system("python pre_process.py --n arrow_15  --w 40 --t train")
os.system("python pre_process.py --n 2tt  --w 150  --d true --i -1")
# os.system("python train/train.py --n 2tt --w 50 --m LSTM")
os.system('python evaluate/evaluate_cate2.py --m LSTM_Embedded_2tt_150_BEST_.h5 --n 2tt --w 150 --c test')
# os.system("python train\keras_train.py --n milan --w 2000")
# X, Y, listActivities = loadDataCase1("cairo", 2000, "train")
# print(listActivities, len(listActivities), set(Y), len(set(Y)))
# import tensorflow as tf
# sequence = [[1], [2, 3], [4, 5, 6]]
# print(tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=2))
# evaluate_model("evaluate/biLSTM-cairo-20210505-032112.h5")


