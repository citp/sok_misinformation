import os
import random
import tensorflow as tf
import argparse
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import (Dense,Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, Embedding, TimeDistributed)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import contextlib 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from DataPrep.Clean_Texts import clean_text

import numpy
import nltk
nltk.download('punkt', quiet = True)

import matplotlib.pyplot as plt
import nltk
import matplotlib.pyplot as plt

plt.style.use('ggplot')
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--test_source', type=str, choices=['isot', 'fakes'], required=True,
                    help='Choose test source: "isot" or "fakes"')

parser.add_argument('--rand_seed', type=int, required=False, help = 'enter integer random seed')

args = parser.parse_args()

np.random.seed(args.rand_seed)
tf.random.set_seed(args.rand_seed)
random.seed(args.rand_seed)


if args.test_source == 'isot':
    orig_file = 'ISOT.csv'
elif args.test_source == 'fakes':
    orig_file = 'fakes.csv'

dataset = pd.read_csv(orig_file)

print(args.test_source)
print(dataset.shape)

texts=[]
texts=dataset['text']
label=dataset['label']

#test_texts = test_set['text']
#test_labels = test_set['label']

labelEncoder=LabelEncoder()
encoded_label=labelEncoder.fit_transform(label)
y=np.reshape(encoded_label,(-1,1))

training_size=int(0.8*dataset.shape[0]) # 80% training set
# print(dataset.shape[0],training_size)
data_train=dataset[:training_size]['text']  
data_rest=dataset[training_size:]['text'] # 20% test set

y_train=y[:training_size]
y_test=y[training_size:]

MAX_SENT_LENGTH = 100
MAX_SENTS = 20
MAX_NB_WORDS = 400000
EMBEDDING_DIM = 100
#VALIDATION_SPLIT = 0.2

vocabulary_size = 400000
time_step=300
embedding_size=100
# Convolution
filter_length = 3
nb_filters = 128
n_gram=5
cnn_dropout=0.0
nb_rnnoutdim=300
rnn_dropout=0.0
nb_labels=1
dense_wl2reg=0.0
dense_bl2reg=0.0


texts=data_train

texts=texts.map(lambda x: clean_text(x))

tokenizer=Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(texts)
encoded_train=tokenizer.texts_to_sequences(texts=texts)
vocab_size_train = len(tokenizer.word_index) + 1
print(vocab_size_train)

x_train = pad_sequences(encoded_train, maxlen=time_step,padding='post')


texts=data_rest

texts=texts.map(lambda x: clean_text(x))


encoded_test=tokenizer.texts_to_sequences(texts=texts)

x_test = pad_sequences(encoded_test, maxlen=time_step,padding='post')


GLOVE_DIR = "."
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding='utf-8')
embeddings_train={}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_train[word] = coefs
f.close()

# print('Total %s word vectors.' % len(embeddings_train))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size_train, embedding_size))
for word, i in tokenizer.word_index.items():
	embedding_vector_train = embeddings_train.get(word)
	if embedding_vector_train is not None:
		embedding_matrix[i] = embedding_vector_train



model = Sequential()
model.add(Embedding(vocab_size_train, embedding_size, input_length=time_step,
                    weights=[embedding_matrix],trainable=False))
model.add(Conv1D(filters=nb_filters,
                 kernel_size=n_gram,
                 padding='valid',
                 activation='relu'))
if cnn_dropout > 0.0:
    model.add(Dropout(cnn_dropout))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(nb_rnnoutdim))
if rnn_dropout > 0.0:
    model.add(Dropout(rnn_dropout))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=5, batch_size=64, validation_data=(x_test, y_test))

print("RANDOM SEED: ", args.rand_seed)

score=model.evaluate(x_test,y_test,verbose=0)
# print('acc: '+str(score[1]))
print(score)

from sklearn.metrics import precision_recall_fscore_support,classification_report
y_pred=model.predict(x_test)
classes_x = np.argmax(y_pred,axis=1)
# predict_x=model.predict(X_test) 
# classes_x=np.argmax(predict_x,axis=1)

bin_labels = []
for i in range(len(y_pred)):
    if y_pred[i] >= 0.5:
        bin_labels = np.append(bin_labels, 1)
    else:
        bin_labels = np.append(bin_labels, 0)


# print(classes_x)
np.savetxt("y_classif_isot_scrubbed4_dummy.csv", y_pred, delimiter=",")
np.savetxt("y_bin_classif_isot_scrubbed4_dummy.csv", bin_labels, delimiter=",")


# print('Classification report:\n',precision_recall_fscore_support(y_test,bin_labels))

# print('Classification report:\n')

fpr, tpr, thresh = metrics.roc_curve(y_test, bin_labels)

# print(fpr, tpr, thresh)

fpr, fnr, thresh = metrics.det_curve(y_test, bin_labels)

print(classification_report(y_test, bin_labels))

print("FPR: ",fpr)
print("FNR: ", fnr)
# print("threshold: ", thresh)

# accuracy = accuracy_score(y_test, bin_labels)
# precision = precision_score(y_test, bin_labels)
# recall = recall_score(y_test, bin_labels)
# f1 = f1_score(y_test, bin_labels)
# auc = roc_auc_score(y_test, bin_labels)
