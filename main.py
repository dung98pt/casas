import os 
from data import loadDataCase1, loadDataCase2
# os.system("python pre_process.py --w1 2000 --w2 100")


X, Y, dictActivities, listActivities = loadDataCase2("cairo", 100, "train")
print(listActivities, len(listActivities))
X, Y, dictActivities, listActivities = loadDataCase2("cairo", 100, "test")
print(listActivities, len(listActivities), set(Y))