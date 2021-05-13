from keras.layers import Layer
from keras.layers import Dense, LSTM, Input, Embedding
from keras.models import Sequential, Model
import keras
import keras.backend as K

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

def model(trainx, trainy, vocab_length):
    nb_classes = trainy.shape[1]
    s1 = trainx.shape[1]
    inputs= Input((s1,))
    x = Embedding(input_dim=vocab_length+1, output_dim=64, input_length=s1)(inputs)
    att_in = LSTM(64, return_sequences=True)(x) # dropout=0.3, recurrent_dropout=0.2
    att_out= attention()(att_in)
    outputs= Dense(32, activation='softmax', trainable=True)(att_out)
    outputs= Dense(nb_classes, activation='softmax', trainable=True)(att_out)
    model= Model(inputs, outputs,  name="Attention")
    model.summary()
    return model