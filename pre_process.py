#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import pickle
from utilities import save_dict
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utilities import load_dict
import json, io 
from datetime import datetime, date

def save_word_id(path_save, tokenizer_):
    tokenizer_json = tokenizer_.to_json()
    with io.open(path_save, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

def load_raw_dataset(input_file):
    df = pd.read_csv(input_file,sep="\t",header=None,names=["last_updated", "date","time","sensor","value","activity"])
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    df = df.set_index("last_updated")
    return df

def save_activity_dict(input_file, dictActivities, dataset_name):
    # lưu vào đường dẫn ./datasets/activities_dictionary
	filename = "./datasets/activities_dictionary/{}_activity_list.pickle".format(dataset_name)
	pickle_out = open(filename,"wb")
	pickle.dump(dictActivities, pickle_out)
	pickle_out.close()

def sequencesToSentences(df2, win_size):
    sequences = []
    labels = []
    for i in range(len(df2)-win_size):
        sequence = " ".join(list((df2["sensor"].iloc[i:i+win_size]+df2["value"][i:i+win_size]).values))
        sequences.append(sequence)
        if len(set(df2["activity"].iloc[i:i+win_size]))==1:
            label = df2["activity"].iloc[i]
        else:
            label = df2["activity"].iloc[i] + "_" + df2["activity"].iloc[i+win_size-1]
        labels.append(label)
    return sequences, labels

no_indexing = []
def indexing_sequence(word_id, sequence):
    idexed_sequence = []
    for i in sequence.split():
        try:
            idexed_sequence.append(word_id[i])
        except:
            idexed_sequence.append(0)
            no_indexing.append(i)
    return idexed_sequence

if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--n', dest='dataset_name', action='store', default='cairo', help='input', required = True)
    p.add_argument('--w', dest='winSize', action='store', default='', help='input', required = True)
    args = p.parse_args()

    #========================
    # START
    #========================
    input_file = "./datasets/{}/data".format(args.dataset_name)
    print("STEP 1: Load dataset + SPLIT DATA")
    df = load_raw_dataset(input_file)
    print(df.head())

    #========================
    # WORD INDEXING DICTIONARY
    #========================
    words = set((df["sensor"].astype('str')+df["value"].astype('str')).values)
    word_index = {}
    for i, word in enumerate(words):
        word_index[word] = i+1

    if not os.path.isdir("./datasets/word_id"):
        os.makedirs("./datasets/word_id")
    save_dict("./datasets/word_id/{}.pickle".format(args.dataset_name), word_index)
    print("len_word_index: ", len(word_index), word_index)

    #========================
    # SPLIT DATA
    #========================
    print("STEP 2: SPLIT DATA")
    split_date = datetime(2021, 5, 19, 8, 30, 0, 0)
    df_train = df[df.index<split_date]
    df_test  =  df[df.index>=split_date]
    df_train = df_train.astype('str')
    df_test  = df_test.astype('str')
    

    #========================
    # WORD INDEXING TRAIN + LABEL DICTIONARY
    #========================
    print("STEP 3: WORD INDEXING TRAIN + LABEL DICTIONARY")
    dict_activities = {}
    train_sentences, train_label_sentences = sequencesToSentences(df_train, int(args.winSize))
    # create label dictionary
    for i, activity in enumerate(set(train_label_sentences)):
        dict_activities[activity] = i
    save_activity_dict(args.dataset_name, dict_activities, args.dataset_name)
    # convert categoriy
    train_label_sentences = [dict_activities[i] for i in train_label_sentences]
    print(dict_activities)
    print("n_sample: ", len(train_sentences), "n_activity: ", len(set(train_label_sentences)))  
    for l in list(set(train_label_sentences)):
        print(l, train_label_sentences.count(l))
    
    # indexing trainset
    indexed_train_sentences = [indexing_sequence(word_index, i) for i in train_sentences]
    print("TRAIN SET:", len(indexed_train_sentences), len(train_label_sentences))
    #========================
    # WORD INDEXING TEST + LABEL DICTIONARY
    #========================
    test_sentences, test_label_sentences = sequencesToSentences(df_test, int(args.winSize))
    test_label_sentences = [dict_activities[i] for i in test_label_sentences]
    indexed_test_sentences = [indexing_sequence(word_index, i) for i in test_sentences]
    print("TEST SET:", len(indexed_test_sentences), len(test_label_sentences))
    print(no_indexing)

    #========================
    # SAVE TO NPY
    ========================
    np.save("./datasets/processed_data/{}_{}_X_train.npy".format(args.dataset_name, args.winSize), np.array(indexed_train_sentences))
    np.save("./datasets/processed_data/{}_{}_Y_train.npy".format(args.dataset_name, args.winSize), train_label_sentences)
    np.save("./datasets/processed_data/{}_{}_X_test.npy".format( args.dataset_name, args.winSize), np.array(indexed_test_sentences))
    np.save("./datasets/processed_data/{}_{}_Y_test.npy".format( args.dataset_name, args.winSize), test_label_sentences)