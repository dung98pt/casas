#!/usr/bin/env python3

from keras.layers import Dense, LSTM, Bidirectional, concatenate, Input
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from tensorflow.keras import layers

def get_LSTM(input_dim, output_dim, max_lenght, no_activities):
    print(input_dim, output_dim)
    model = Sequential(name='LSTM')
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model.add(LSTM(output_dim))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_biLSTM(input_dim, output_dim, max_lenght, no_activities):
    print(input_dim, output_dim)
    model = Sequential(name='biLSTM')
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model.add(Bidirectional(LSTM(output_dim)))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_Ensemble2LSTM(input_dim, output_dim, max_lenght, no_activities):
    print(input_dim, output_dim)
    # model1 = Sequential()
    # model1.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    # model1.add(Bidirectional(LSTM(output_dim)))

    # model2 = Sequential()
    # model2.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    # model2.add(LSTM(output_dim))

    # model = Sequential(name='Ensemble2LSTM')
    # model.add(concatenate([model1, model2]))
    # model.add(Dense(no_activities, activation='softmax'))
    first_input = Input(shape=(max_lenght, ))
    second_input = Input(shape=(max_lenght, ))
    embed1 = Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True)(first_input)
    embed2 = Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True)(second_input)
    lstm2 = LSTM(output_dim)(embed2)
    bidirec1 = Bidirectional(layers.LSTM(output_dim))(embed1)
    concat = concatenate([bidirec1, lstm2])
    out = Dense(no_activities, activation='softmax')(concat)
    model = Model(inputs=[first_input, second_input], outputs=out, name='Ensemble2LSTM')
    return model


def get_CascadeEnsembleLSTM(input_dim, output_dim, max_lenght, no_activities):
    # model1 = Sequential()
    # model1.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    # model1.add(Bidirectional(LSTM(output_dim, return_sequences=True)))

    # model2 = Sequential()
    # model2.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    # model2.add(LSTM(output_dim, return_sequences=True))

    # model = Sequential(name='CascadeEnsembleLSTM')
    # model.add(concatenate([model1, model2]))
    # model.add(LSTM(output_dim))
    # model.add(Dense(no_activities, activation='softmax'))
    first_input = Input(shape=(max_lenght, ))
    second_input = Input(shape=(max_lenght, ))
    embed1 = Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True)(first_input)
    embed2 = Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True)(second_input)
    bi1 = Bidirectional(layers.LSTM(output_dim, return_sequences=True))(embed1)
    lstm2 = LSTM(output_dim, return_sequences=True)(embed2)
    concat = concatenate([bi1, lstm2])
    concated_lstm = LSTM(output_dim)(concat)
    out = Dense(no_activities, activation='softmax')(concated_lstm)
    model = Model(inputs=[first_input, second_input], outputs=out, name='CascadeEnsembleLSTM')
    return model


def get_CascadeLSTM(input_dim, output_dim, max_lenght, no_activities):
    model = Sequential(name='CascadeLSTM')
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model.add(Bidirectional(LSTM(output_dim, return_sequences=True)))
    model.add(LSTM(output_dim))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def compileModel(model):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def get_model(args_model, input_dim, no_activities, max_lenght, units = 64):
  model = None
  if args_model == 'LSTM':
    model = get_LSTM(input_dim, units, max_lenght, no_activities)
  elif args_model == 'biLSTM':
    model = get_biLSTM(input_dim, units, max_lenght, no_activities)
  elif args_model == 'Ensemble2LSTM':
    model = get_Ensemble2LSTM(input_dim, units, max_lenght, no_activities)
  elif args_model == 'CascadeEnsembleLSTM':
    model = get_CascadeEnsembleLSTM(input_dim, units, max_lenght, no_activities)
  elif args_model == 'CascadeLSTM':
    model = get_CascadeLSTM(input_dim, units, max_lenght, no_activities)
  model = compileModel(model)
  return model
