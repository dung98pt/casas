import os 
import numpy as np
from data import loadDataCase1, loadDataCase2
from sklearn.utils import compute_class_weight
# os.system("python pre_process.py --w1 2000 --w2 100")
# X, Y, listActivities = loadDataCase1("cairo", 2000, "train")
# print(listActivities, len(listActivities), set(Y), len(set(Y)))



X, Y, listActivities = loadDataCase1("cairo", 2000, "test")
class_weight = compute_class_weight('balanced', np.unique(Y),Y) 
Y = list(Y)
n = len(set(Y))
for i in set(Y):
    print(i, len(Y)/n/Y.count(i))

print(class_weight)