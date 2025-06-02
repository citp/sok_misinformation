import os

import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import (Dense,Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, Embedding, TimeDistributed)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

# from keras.layers import LSTM,Conv1D
# from keras.layers import MaxPooling1D
# from keras.layers import Flatten
# from keras.layers import Embedding
# from keras.preprocessing import sequence
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
# from keras import optimizers
# from keras.layers import TimeDistributed

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
# from keras.regularizers import l2
from DataPrep.Clean_Texts import clean_text

#import numpy
#seed = 1
#numpy.random.seed(seed)

import nltk
nltk.download('punkt')

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


# dataset = pd.read_csv('syrian.csv')

# test data
test_set = pd.read_csv('real_reuters.csv')

# training data
dataset = pd.read_csv('ISOT.csv')
print(dataset.shape)

texts=[]
texts=dataset['text']#####################################
label=dataset['label']

texts = np.array(texts)
label = np.array(label)

test_texts = test_set['text']
test_labels = test_set['label']

test_texts = np.array(test_texts)
test_labels = np.array(test_labels)

labelEncoder=LabelEncoder()
encoded_label=labelEncoder.fit_transform(label)
y=np.reshape(encoded_label,(-1,1))


#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# Age    Sex       Disease
# ----  ------ |  ---------
  
#   X_train    |   y_train   )
#                            )
#  5       F   |  A Disease  )
#  15      M   |  B Disease  ) 
#  23      M   |  B Disease  ) training
#  39      M   |  B Disease  ) data
#  61      F   |  C Disease  )
#  55      M   |  F Disease  )
#  76      F   |  D Disease  )
#  88      F   |  G Disease  )
# -------------|------------
   
#   X_test     |    y_test

#  63      M   |  C Disease  )
#  46      F   |  C Disease  ) test
#  28      M   |  B Disease  ) data
#  33      F   |  B Disease  )



training_size=int(0.8*dataset.shape[0]) # 80% training set
print(dataset.shape[0],training_size)
data_train=dataset[:training_size]['text']  
data_rest=dataset[training_size:]['text'] # 20% test set

y_train=y[:training_size]
#y_test=y[training_size:]

x_test_data = test_texts
y_test = test_labels

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

# texts=texts.map(lambda x: clean_text(x))
texts = pd.Series(texts).map(clean_text)

tokenizer=Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(texts)
encoded_train=tokenizer.texts_to_sequences(texts=texts)
vocab_size_train = len(tokenizer.word_index) + 1
print(vocab_size_train)

#x_train = sequence.pad_sequences(encoded_train, maxlen=time_step,padding='post')
x_train = pad_sequences(encoded_train, maxlen=time_step,padding='post')


#texts=data_rest

texts = x_test_data

#texts=texts.map(lambda x: clean_text(x))
texts = pd.Series(texts).map(clean_text)

encoded_test=tokenizer.texts_to_sequences(texts=texts)

#x_test = sequence.pad_sequences(encoded_test, maxlen=time_step,padding='post')
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

print('Total %s word vectors.' % len(embeddings_train))

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

score=model.evaluate(x_test,y_test,verbose=1)
print('acc: '+str(score[1]))

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


#print('Classification report:\n',classification_report(y_test,y_pred))
print(classes_x)
# np.savetxt("y_classif_syrian_abr.csv", y_pred, delimiter=",")
# np.savetxt("y_bin_classif_syrian_abr.csv", bin_labels, delimiter=",")

np.savetxt("y_classif_reuters_ISOT_real_rerun.csv", y_pred, delimiter=",")
np.savetxt("y_bin_reuters_ISOT_real_rerun.csv", bin_labels, delimiter=",")


#print('Classification report:\n',classification_report(y_test,y_pred))

plot_history(history)

#print('Classification report:\n',precision_recall_fscore_support(y_test,y_pred))
#print(y_pred)
