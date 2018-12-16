import numpy as np
import collections
import pickle
import argparse

from keras import Input
from keras.engine import Model
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers.core import Activation, Dense, RepeatVector
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.layers.wrappers import TimeDistributed

class Config:
    def __init__(self):
        self.UNKNOWN_TOKEN = "**UNK**"
        self.PAD_TOKEN = "**PAD**"
        self.INPUT_VOCAB_SIZE = 4096
        self.OUTPUT_VOCAB_SIZE = 60000
        self.N_NEIGHBORS = 10
        self.KTH_COMMON = 1

        self.SEQ_LEN = 5
        self.HIDDEN_LAYER_SIZE = 80
        self.HIDDEN_LAYER_SIZE2 = 3500
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 5
        self.ACCURACY = 0.995
        self.ACCURACY2 = 0.9
        self.PLATEAU_LEN = 1
        self.CHUNK_SIZE1 = 25000
        self.CHUNK_SIZE2 = 20

        self.PROCESSED_FILE="vocab.pkl"
        self.TRAINING_FILE = "training.csv"  # space separated
        self.EVAL_FILE = "eval.csv"  # space separated
        self.MODEL_FILE = "model.h5"
        self.CONFIG_FILE = "config.json"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', action='store_true', default=False,
                        dest='is_pload',
                        help='Load processed training and validation data.  Sets -i and -o options.')
    parser.add_argument('-i', action='store_true', default=False,
                        dest='is_iload',
                        help='Load input vocabulary')
    parser.add_argument('-o', action='store_true', default=False,
                        dest='is_oload',
                        help='Load output vocabulary')
    parser.add_argument('-a', action='store_true', default=False,
                        dest='load_model1',
                        help='Load encoder model from file')
    parser.add_argument('-b', action='store_true', default=False,
                        dest='load_model2',
                        help='Load LSTM model from file')

    return parser.parse_args()

def generate_sequence_for_lstm(encoder, arr, input_vocab_size, output_vocab_size):
    c = 0
    input = arr[0]
    output = arr[1]
    length = len(input)
    indices = list(range(length))
#    np.random.shuffle(indices)
    input = input[indices,:]
    output = output[indices]

    while True:
        if (c + config.CHUNK_SIZE2 > length):
            np.random.shuffle(indices)
            input = input[indices, :]
            output = output[indices]
            c = 0
        i_slice = input[c:c+config.CHUNK_SIZE2,:] #(50,20) (50,20,4098)
        i_slice = np_utils.to_categorical(i_slice.reshape([-1]), num_classes=input_vocab_size).reshape([-1,config.N_NEIGHBORS,input_vocab_size])
        i_encoded_slice = encoder.predict(i_slice)
        i_slice = i_encoded_slice.reshape([-1,config.SEQ_LEN,config.HIDDEN_LAYER_SIZE])#(20,5,80)
        o_slice = output[c:c+config.CHUNK_SIZE2]
#        o_slice = np_utils.to_categorical(o_slice, num_classes=output_vocab_size)#(20, 60002)
        c = c + config.CHUNK_SIZE2
        print("i_slice.shape = {} o_slice.shape {}".format(i_slice.shape, o_slice.shape))
        yield (i_slice, o_slice)

def get_generators_for_lstm(encoder, training_arr, validation_arr, input_vocab_size, output_vocab_size):
    training_generator = generate_sequence_for_lstm(encoder, training_arr, input_vocab_size, output_vocab_size)
    eval_generator = generate_sequence_for_lstm(encoder, validation_arr, input_vocab_size, output_vocab_size)
    return (training_generator, eval_generator)

if __name__ == "__main__":
    config = Config()

    results = parse_args()

    with open('p_4096_60000_vocab.pkl', 'rb') as f:
        (training_arr, validation_arr) = pickle.load(f)
    with open('i_4096_vocab.pkl', 'rb') as f1:
        i_map = pickle.load(f1)
    with open('o_60000_vocab.pkl', 'rb') as f2:
        o_map = pickle.load(f2)
    autoencoder = load_model("autoencoder.4096_80.model.h5")
    encoder = load_model("encoder.4096_80.model.h5")


    training_generator, eval_generator = get_generators_for_lstm(encoder, training_arr, validation_arr,
                                                                 i_map[0], o_map[0])

    n_chunks = int(len(training_arr[0]) / config.CHUNK_SIZE2)

    train_data = next(training_generator)
    eval_data = next(eval_generator)


    embedding = load_model("embedding_3500_60000.model.h5")
    lstm = load_model("lstm_3500_60000.model.h5")

    for a in lstm.predict(eval_data[0], batch_size=1).argmax(axis=1):
        print(o_map[2][a])
    for b in eval_data[1]:
        print(o_map[2][b])