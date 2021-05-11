import pickle

def load_dict(path_load):
    pickle_in = open(path_load,"rb")
    dictActivities = pickle.load(pickle_in)
    return dictActivities

def save_dict(path_save, dict):
	pickle_out = open(path_save, "wb")
	pickle.dump(dict, pickle_out)
	pickle_out.close()

def label_encode(labels, tool=False):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    print(label_encoder.get_params())
    labels = label_encoder.fit_transform(labels)
    if tool:
        return labels, label_encoder
    else:
        return labels