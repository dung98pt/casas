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
	df = pd.read_csv(input_file,sep="\t",header=None,names=["date","time","sensor","value","activity","log"])
	return df

def clean_and_prepare(df):
	#rempli les valeurs NaN de la colonne log avec la valeur précédente
	df.log = df.log.fillna(method='ffill')
	#rempli les valeur NaN de la colonne activity avec lavaleur de la colonne log
	df['activity'] = df['activity'].fillna(df['log'])
	df['activity'] = df['activity'].replace("end", "Other")
	df['activity'] = df['activity'].fillna("Other")
	df['activity'] = df['activity'].replace("begin", None)
	return df

def save_activity_dict(input_file, dictActivities, dataset_name):
    # lưu vào đường dẫn ./datasets/activities_dictionary
	filename = "./datasets/activities_dictionary/{}_activity_list.pickle".format(dataset_name)
	pickle_out = open(filename,"wb")
	pickle.dump(dictActivities, pickle_out)
	pickle_out.close()

def generate_sentence(df2):
    sentence = ""
    val = "" 
    sensors = df2.sensor.values
    values = df2.value.values
    #iterate on sensors list
    for i in range(len(sensors)):
        val = values[i]
        if i == len(sensors) - 1:
            sentence += "{}{}".format(sensors[i],val)
        else:
            sentence += "{}{} ".format(sensors[i],val)
    return sentence

def segment_activities(df):
	activitiesSeq = []
	ponentialIndex = df.activity.ne(df.activity.shift())
	ii = np.where(ponentialIndex == True)[0]
	for i,end in enumerate(ii):
	    if i > 0 :
	        dftmp = df[ii[i-1]:end]
	        activitiesSeq.append(dftmp)
	return activitiesSeq

def slidingWindow(sequence,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    numOfChunks = int(((len(sequence)-winSize)/step)+1)
    # Do the work
    if winSize > len(sequence):
        yield sequence[0:len(sequence)]
    else:
        for i in range(0,numOfChunks*step,step):
            yield sequence[i:i+winSize]

def sequencesToSentences(activitySequences):
	sentences = []
	label_sentences = []
	for i in range(len(activitySequences)):
		sentence = generate_sentence(activitySequences[i])
		sentences.append(sentence)
		label_sentences.append(activitySequences[i].activity.values[0])
	return sentences, label_sentences

def padding_sequence(sequence, winSize):
    if winSize > len(sequence):
        return sequence[0:len(sequence)]
    else:
        return sequence[:winSize]
    # cho xử lý kiểu 1
if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--n', dest='dataset_name', action='store', default='cairo', help='input', required = False)
    p.add_argument('--t', dest='cut_test', action='store', default=True, help='input', required = True)
    p.add_argument('--w1', dest='winSize_1', action='store', default='', help='input', required = False)
    p.add_argument('--w2', dest='winSize_2', action='store', default='', help='input', required = False)
    args = p.parse_args()
    input_file = "./datasets/{}/data".format(args.dataset_name)
    print("STEP 1: Load dataset")
    df = load_raw_dataset(input_file)
    print("STEP 2: prepare dataset")
    df = clean_and_prepare(df)
    ## Segment dataset in sequence of activity ##
    print("STEP 3: segment dataset in sequence of activity")
    activitySequences = segment_activities(df)
    ## Transform sequences of activity in sentences ##
    print("STEP 4: transform sequences of activity in sentences")
    sentences, label_sentences = sequencesToSentences(activitySequences)
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
    # for i in list(set(label_sentences)):
        # print(i, label_sentences.count(i), label_sentences.count(i)*100/n_sample)
    print("n_sample: ", n_sample, n_activity)  
    # num_test_sample = 100
    # for i in range(n_sample-num_test_sample):
    #     test_sample = label_sentences[i:i+num_test_sample]
    #     if len(set(test_sample)) < n_activity:
    #         continue
    #     else:
    #         scores = []
    #         for j in list(set(label_sentences)):
    #             scores.append(test_sample.count(j)*100/num_test_sample)
    #         print("===================================>>>   ", i, np.mean(scores))

    if args.cut_test=="False":
        ## Indexization ##
        print("STEP 5: sentences indexization")
        tokenizer = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(sentences)
        print("tokenizer: ", tokenizer)
        word_index = tokenizer.word_index
        # print(word_index)
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
        #   np.save("./datasets/preprocess_raw/{}_X_train.npy".format(args.dataset_name), np.array(indexed_sentences_train))
        #   np.save("./datasets/preprocess_raw/{}_Y_train.npy".format(args.dataset_name), label_sentences_train)
        #   np.save("./datasets/preprocess_raw/{}_X_test.npy".format(args.dataset_name), np.array(indexed_sentences_test))
        #   np.save("./datasets/preprocess_raw/{}_Y_test.npy".format(args.dataset_name), label_sentences_test)
        print("STEP 6: split indexed sentences in sliding windows")
        # Xử lý kiểu 1
        if args.winSize_1 != "":
            print("   --> Xu ly kieu 1")
            winSize_1 = int(args.winSize_1)
            x_train = []
            x_test = []
            for i in indexed_sentences_train:
                x_train.append(padding_sequence(i, winSize_1))
            for i in indexed_sentences_test:
                x_test.append(padding_sequence(i, winSize_1))
            x_train = pad_sequences(x_train)
            x_test = pad_sequences(x_test)
            np.save("./datasets/preprocess_cate1/{}_{}_X_train.npy".format(args.dataset_name, winSize_1), np.array(x_train))
            np.save("./datasets/preprocess_cate1/{}_{}_Y_train.npy".format(args.dataset_name, winSize_1), label_sentences_train)
            np.save("./datasets/preprocess_cate1/{}_{}_X_test.npy".format(args.dataset_name, winSize_1), np.array(x_test))
            np.save("./datasets/preprocess_cate1/{}_{}_Y_test.npy".format(args.dataset_name, winSize_1), label_sentences_test)

        # Xử lý kiểu 2
        if args.winSize_2 != "":
            print("   --> Xu ly kieu 2")
            winSize_2 = int(args.winSize_2)
            # Split in sliding windows ##
            X_windowed_train = []
            Y_windowed_train = []
            X_windowed_test = []
            Y_windowed_test = []
            step = 1
            # train
            for i,s in enumerate(indexed_sentences_train):
                chunks = slidingWindow(s,winSize_2,step)
                for chunk in chunks:
                    if len(chunk)>0:
                        X_windowed_train.append(chunk)
                        Y_windowed_train.append(label_sentences_train[i])
            # test
            for i,s in enumerate(indexed_sentences_test):
                chunks = slidingWindow(s,winSize_2,step)
                for chunk in chunks:
                    if len(chunk)>0:
                        X_windowed_test.append(chunk)
                        Y_windowed_test.append(label_sentences_test[i])
            ## Pad windows ##
            print("STEP 7: pad sliding windows")
            X_windowed_train = pad_sequences(X_windowed_train)
            X_windowed_test = pad_sequences(X_windowed_test)
            ## Save files ##
            np.save("./datasets/preprocess_cate2/{}_{}_X_train.npy".format(args.dataset_name, winSize_2), np.array(X_windowed_train))
            np.save("./datasets/preprocess_cate2/{}_{}_Y_train.npy".format(args.dataset_name, winSize_2), Y_windowed_train)
            np.save("./datasets/preprocess_cate2/{}_{}_X_test.npy".format(args.dataset_name,  winSize_2), np.array(X_windowed_test))
            np.save("./datasets/preprocess_cate2/{}_{}_Y_test.npy".format(args.dataset_name,  winSize_2), Y_windowed_test)