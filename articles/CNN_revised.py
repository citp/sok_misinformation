import os
import random
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Embedding)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from DataPrep.Clean_Texts import clean_text
import matplotlib.pyplot as plt

nltk.download('punkt')
plt.style.use('ggplot')

# set default random seed
SEED = 99

# args 
parser = argparse.ArgumentParser()
parser.add_argument('--test_source', type=str, choices=['nytimes', 'reuters'], required=True,
                    help='Choose test source: "nytimes" or "reuters"')

parser.add_argument('--rand_seed', type=int, required=False, help = 'enter integer random seed')

args = parser.parse_args()

np.random.seed(args.rand_seed)
tf.random.set_seed(args.rand_seed)
random.seed(args.rand_seed)

# constants
vocabulary_size = 400000
time_step = 300
embedding_size = 100
filter_length = 3
nb_filters = 128
n_gram = 5
cnn_dropout = 0.0
nb_rnnoutdim = 300
rnn_dropout = 0.0

# pad all inputs together
def preprocess_texts(texts, tokenizer, maxlen):
    encoded = tokenizer.texts_to_sequences(texts)
    return pad_sequences(encoded, maxlen=maxlen, padding='post', truncating='post')

# load training data
train_dataset = pd.read_csv('ISOT.csv')
texts = train_dataset['text'].map(clean_text)
labels = train_dataset['label']

labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(labels)
y = np.reshape(y, (-1, 1))

tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(texts)
x_train = preprocess_texts(texts, tokenizer, maxlen=time_step)
vocab_size_train = len(tokenizer.word_index) + 1

# load test data from --test_source arg 
if args.test_source == 'nytimes':
    orig_file = 'real_nytimes.csv'
    mod_file = 'modified_nytimes.csv'
elif args.test_source == 'reuters':
    orig_file = 'real_reuters.csv'
    mod_file = 'modified_reuters.csv'

original_df = pd.read_csv(orig_file)
modified_df = pd.read_csv(mod_file)

# clean text
original_df['text'] = original_df['text'].map(clean_text)
modified_df['text'] = modified_df['text'].map(clean_text)

combined_texts = pd.concat([original_df['text'], modified_df['text']], axis=0)
x_combined = preprocess_texts(combined_texts, tokenizer, maxlen=time_step)
x_orig = x_combined[:len(original_df)]
x_mod = x_combined[len(original_df):]

y_test_orig = labelEncoder.transform(original_df['label'])
y_test_mod = labelEncoder.transform(modified_df['label'])
y_test_orig = np.reshape(y_test_orig, (-1, 1))
y_test_mod = np.reshape(y_test_mod, (-1, 1))

# load GloVe embeddings ---
GLOVE_DIR = "."
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((vocab_size_train, embedding_size))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# build model
model = Sequential()
model.add(Embedding(vocab_size_train, embedding_size, input_length=time_step,
                    weights=[embedding_matrix], trainable=False))
model.add(Conv1D(filters=nb_filters, kernel_size=n_gram, activation='relu'))
if cnn_dropout > 0.0:
    model.add(Dropout(cnn_dropout))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(nb_rnnoutdim))
if rnn_dropout > 0.0:
    model.add(Dropout(rnn_dropout))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# train
history = model.fit(x_train, y, epochs=5, batch_size=64, validation_data=(x_orig, y_test_orig), shuffle=False)

print("RANDOM SEED: ", args.rand_seed)

# evaluate orig dataset
print("\nEvaluating ORIGINAL texts:")
score_orig = model.evaluate(x_orig, y_test_orig, batch_size=64, verbose=1)
y_pred_orig = model.predict(x_orig, batch_size=64)
y_bin_orig = (y_pred_orig >= 0.5).astype(int)
print('Original Classification Report:\n')

print(metrics.classification_report(y_test_orig, y_bin_orig))

# fpr, fnr, thresh = metrics.det_curve(y_test_orig, y_bin_orig)
# print(classification_report(y_test_orig, y_bin_orig))

# print("FPR: ",fpr)
# print("FNR: ", fnr)

# evaluate modified dataset
print("\nEvaluating MODIFIED texts:")
score_mod = model.evaluate(x_mod, y_test_mod, batch_size=64, verbose=1)
y_pred_mod = model.predict(x_mod, batch_size=64)
y_bin_mod = (y_pred_mod >= 0.5).astype(int)
print('Modified Classification Report:\n')

print(metrics.classification_report(y_test_mod, y_bin_mod))

fpr, fnr, thresh = metrics.det_curve(y_test_mod, y_bin_mod)
# print(classification_report(y_test_mod, y_bin_mod))

print("FPR: ",fpr)
print("FNR: ", fnr)

# save results ---

np.savetxt(f"y_bin_pred_original_{args.test_source}_{args.rand_seed}.csv", y_bin_orig, delimiter=",")
np.savetxt(f"y_bin_pred_modified_{args.test_source}_{args.rand_seed}.csv", y_bin_mod, delimiter=",")

# optional delta and comparison
# delta_probs = y_pred_mod - y_pred_orig
# np.savetxt(f"delta_probs_{args.test_source}.csv", delta_probs, delimiter=",")

# comparison_df = pd.DataFrame({
#     'text_original': original_df['text'],
#     'text_modified': modified_df['text'],
#     'label_original': y_test_orig.flatten(),
#     'label_modified': y_test_mod.flatten(),
#     'prob_original': y_pred_orig.flatten(),
#     'prob_modified': y_pred_mod.flatten(),
#     'delta_prob': delta_probs.flatten(),
#     'bin_original': y_bin_orig.flatten(),
#     'bin_modified': y_bin_mod.flatten()
# })
# comparison_df.to_csv(f'prediction_comparison_{args.test_source}_{args.rand_seed}.csv', index=False)

# print("\n average probability change on modified vs original:", np.mean(delta_probs))
