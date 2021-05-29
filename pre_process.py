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

def save_word_id(path_save, tokenizer_):
    tokenizer_json = tokenizer_.to_json()
    with io.open(path_save, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

def load_raw_dataset(input_file):
	df = pd.read_csv(input_file,sep="\t",header=None,names=["date","time","sensor","value","activity"])
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
    p.add_argument('--t', dest='test', action='store', default='test', help='input', required = True)
    args = p.parse_args()

    if args.test=="test":
        input_file = "./datasets/{}/test".format(args.dataset_name)
        df = load_raw_dataset(input_file)
        print(df.head())
        print(len(df))
        df = df.astype('str')
        sentences, label_sentences = sequencesToSentences(df, int(args.winSize))
        # load dict_activity
        activities_dict_path = "./datasets/activities_dictionary/{}_activity_list.pickle".format(args.dataset_name)
        dict_activities = load_dict(activities_dict_path)
        word_id_path = "./datasets/word_id/{}.pickle".format(args.dataset_name)
        word_id = load_dict(word_id_path)

        #process
        indexed_sentences = [indexing_sequence(word_id, i) for i in sentences]
        label_sentences = [dict_activities[i] for i in label_sentences]
        print("number of sequence:", len(indexed_sentences), len(label_sentences))
        print(indexed_sentences[0])
        print(label_sentences[:10])
        indexed_sentences = np.array(indexed_sentences)
        t = 0
        for i in indexed_sentences:
            if 0 in i:
                t += 1
        print(t)
        print(set(no_indexing))
        np.save("./datasets/processed_data/{}_{}_X_deploy.npy".format( args.dataset_name, args.winSize), indexed_sentences)
        np.save("./datasets/processed_data/{}_{}_Y_deploy.npy".format( args.dataset_name, args.winSize), label_sentences)
    else:
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
        # tokenizer = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^`{|}~\t\n')
        # tokenizer.fit_on_texts(sentences)
        # word_index = tokenizer.word_index
        words = set((df["sensor"]+df["value"]).values)
        word_index = {}
        for i, word in enumerate(words):
            word_index[word] = i+1

        if not os.path.isdir("./datasets/word_id"):
            os.makedirs("./datasets/word_id")
        # save_word_id("./datasets/word_id/{}.json".format(args.dataset_name), tokenizer)
        save_dict("./datasets/word_id/{}.pickle".format(args.dataset_name), word_index)
        print("len_word_index: ", len(word_index), word_index)

        # indexed_sentences = tokenizer.texts_to_sequences(sentences)
        indexed_sentences = [indexing_sequence(word_index, i) for i in sentences]
        print(set([len(i.split()) for i in sentences]))
        print(set([len(i) for i in indexed_sentences]))
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