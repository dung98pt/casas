import os 
from data import loadDataCase1, loadDataCase2
# os.system("python pre_process.py --w1 2000 --w2 100")
X, Y, listActivities = loadDataCase1("cairo", 2000, "train")
print(listActivities, len(listActivities), set(Y), len(set(Y)))

X, Y, listActivities = loadDataCase1("cairo", 2000, "test")
print(listActivities, len(listActivities), set(Y), len(set(Y)))