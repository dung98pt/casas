#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import pickle
from utilities import save_dict
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_raw_dataset(input_file):
	df = pd.read_csv(input_file,sep="\t",header=None,names=["date","time","sensor","value","activity"])
	return df

def save_activity_dict(input_file, dictActivities, dataset_name):
    # lưu vào đường dẫn ./datasets/activities_dictionary
	filename = "./datasets/activities_dictionary/{}_activity_list.pickle".format(dataset_name)
	pickle_out = open(filename,"wb")
	pickle.dump(dictActivities, pickle_out)
	pickle_out.close()

def segment_activities(df):
	activitiesSeq = []
	ponentialIndex = df.activity.ne(df.activity.shift())
	ii = np.where(ponentialIndex == True)[0]
	for i,end in enumerate(ii):
	    if i > 0 :
	        dftmp = df[ii[i-1]:end]
	        activitiesSeq.append(dftmp)
	return activitiesSeq

def sequencesToSentences(df2, win_size):
    sequences = []
    labels = []
    for i in range(len(df2)-win_size):
        sequence = " ".join(list((df2["sensor"].iloc[i:i+win_size]+df["value"][i:i+win_size]).values))
        sequences.append(sequence)
        if len(set(df2["activity"][i:i+win_size]))==1:
            label = df2["activity"][i]
        else:
            label = df2["activity"][i] + "_" + df2["activity"][i+win_size-1]
        labels.append(label)
    return sequences, labels

def padding_sequence(sequence, winSize):
    if winSize > len(sequence):
        return sequence[0:len(sequence)]
    else:
        return sequence[:winSize]
    # cho xử lý kiểu 1
if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--n', dest='dataset_name', action='store', default='cairo', help='input', required = True)
    p.add_argument('--w', dest='winSize', action='store', default='', help='input', required = True)
    args = p.parse_args()
    input_file = "./datasets/{}/data".format(args.dataset_name)
    print("STEP 1: Load dataset")
    df = load_raw_dataset(input_file)
    df = df.astype('str')

    ## Transform sequences of activity in sentences ##
    print("STEP 4: transform sequences of activity in sentences")
    sentences, label_sentences = sequencesToSentences(df, int(args.winSize))
    """
    Đoạn này vẫn theo thứ tự nè, xử đẹp từ đây nha
    """
    dict_activities = {}
    for i, activity in enumerate(set(label_sentences)):
        dict_activities[activity] = i
    save_activity_dict(args.dataset_name, dict_activities, args.dataset_name)
    print(dict_activities)
    label_sentences = [dict_activities[i] for i in label_sentences]
    n_sample = len(sentences)
    n_activity = len(set(label_sentences))
    print("n_sample: ", n_sample, "n_activity: ", n_activity)  
    for l in list(set(label_sentences)):
        print(l, label_sentences.count(l))
        
    print("STEP 5: sentences indexization")
    # word indexing và lưu mấy file linh tinh
    tokenizer = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    if not os.path.isdir("./datasets/word_id"):
        os.makedirs("./datasets/word_id")
    save_dict("./datasets/word_id/{}.pickle".format(args.dataset_name), word_index)
    indexed_sentences = tokenizer.texts_to_sequences(sentences)
    print("number of sequence:", len(indexed_sentences), len(label_sentences))
    #==============================
    #     phải cắt từ chỗ này
    #==============================
    indexed_sentences_train, indexed_sentences_test, label_sentences_train, label_sentences_test = train_test_split(indexed_sentences, label_sentences, test_size=0.2, random_state=7, stratify=label_sentences)
    print("label_sentences_test", len(label_sentences_test), len(set(label_sentences_test)))
    np.save("./datasets/processed_data/{}_{}_X_train.npy".format(args.dataset_name, args.winSize), np.array(indexed_sentences_train))
    np.save("./datasets/processed_data/{}_{}_Y_train.npy".format(args.dataset_name, args.winSize), label_sentences_train)
    np.save("./datasets/processed_data/{}_{}_X_test.npy".format( args.dataset_name, args.winSize), np.array(indexed_sentences_test))
    np.save("./datasets/processed_data/{}_{}_Y_test.npy".format( args.dataset_name, args.winSize), label_sentences_test)
