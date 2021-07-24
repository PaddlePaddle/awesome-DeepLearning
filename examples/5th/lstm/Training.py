import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pickle
import numpy as np
import string
import os
import glob
import re



lines = []
for i in glob.glob("dataset/*.txt"):
  file = open(i, "r", encoding = "utf8")
  for j in file:
      lines.append(j)

data = ""

for i in lines:
    data = ' '. join(lines)
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('\u200c', '').replace('\u200d','').replace("'","").replace(".","").replace(":","").replace(",","").replace("@","").replace("%","")
# data = re.sub(r'[A-Za-z]*[0-9]*',"",data)


translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
new_data = data.translate(translator)

z = []

for i in new_data.split():
    if i not in z:
        z.append(i)
        
data = ' '.join(z)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
sequence_data = tokenizer.texts_to_sequences([data])[0]
vocab_size = len(tokenizer.word_index) + 1

sequences = []

for i in range(1, len(sequence_data)):
    words = sequence_data[i-1:i+1]
    sequences.append(words)
    
print("The Length of sequences are: ", len(sequences))

sequences = np.array(sequences)
X = []
y = []

for i in sequences:
    X.append(i[0])
    y.append(i[1])
    
X = np.array(X)
y = np.array(y)


print("The Request is: ", X[:10])
print("The Responses are: ", y[:10])

y = to_categorical(y, num_classes=vocab_size)


model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation="softmax"))
model.summary()

checkpoint = ModelCheckpoint("next_word.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto')

reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)


model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))
model.fit(X, y, epochs=500, batch_size=64, callbacks=[checkpoint, reduce])



