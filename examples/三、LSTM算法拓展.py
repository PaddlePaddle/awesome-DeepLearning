{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Next Word Prediction:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Importing The Required Libraries:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "from tensorflow.keras.layers import Embedding, LSTM, Dense\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "import pickle\n",
    "import numpy as np\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The First Line:  ﻿One morning, when Gregor Samsa woke from troubled dreams, he found\n",
      "\n",
      "The Last Line:  first to get up and stretch out her young body.\n"
     ]
    }
   ],
   "source": [
    "\"\"\"\n",
    "    Dataset: http://www.gutenberg.org/cache/epub/5200/pg5200.txt\n",
    "    Remove all the unnecessary data and label it as Metamorphosis-clean.\n",
    "    The starting and ending lines should be as follows.\n",
    "\n",
    "\"\"\"\n",
    "\n",
    "\n",
    "file = open(\"metamorphosis_clean.txt\", \"r\", encoding = \"utf8\")\n",
    "lines = []\n",
    "\n",
    "for i in file:\n",
    "    lines.append(i)\n",
    "    \n",
    "print(\"The First Line: \", lines[0])\n",
    "print(\"The Last Line: \", lines[-1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Cleaning the data:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin.  He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections.  The bedding was hardly able to cover it and seemed ready to slide off any moment.'"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = \"\"\n",
    "\n",
    "for i in lines:\n",
    "    data = ' '. join(lines)\n",
    "    \n",
    "data = data.replace('\\n', '').replace('\\r', '').replace('\\ufeff', '')\n",
    "data[:360]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'One morning  when Gregor Samsa woke from troubled dreams  he found himself transformed in his bed into a horrible vermin   He lay on his armour like back  and if he lifted his head a little he could see his brown belly  slightly domed and divided by arches into stiff sections   The bedding was hardly able to cover it and seemed ready to slide off any moment   His many legs  pitifully thin compared with the size of the rest of him  waved about helplessly as he looked    What s happened to me   he'"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import string\n",
    "\n",
    "translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space\n",
    "new_data = data.translate(translator)\n",
    "\n",
    "new_data[:500]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. He lay on armour-like back, and if lifted head little could see brown belly, slightly domed divided by arches stiff sections. The bedding was hardly able to cover it seemed ready slide off any moment. His many legs, pitifully thin compared with the size of rest him, waved about helplessly as looked. \"What\\'s happened me?\" thought. It wasn\\'t dream. room, proper human room altho'"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z = []\n",
    "\n",
    "for i in data.split():\n",
    "    if i not in z:\n",
    "        z.append(i)\n",
    "        \n",
    "data = ' '.join(z)\n",
    "data[:500]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Tokenization:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[17, 53, 293, 2, 18, 729, 135, 730, 294, 8]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tokenizer = Tokenizer()\n",
    "tokenizer.fit_on_texts([data])\n",
    "\n",
    "# saving the tokenizer for predict function.\n",
    "pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))\n",
    "\n",
    "sequence_data = tokenizer.texts_to_sequences([data])[0]\n",
    "sequence_data[:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2617\n"
     ]
    }
   ],
   "source": [
    "vocab_size = len(tokenizer.word_index) + 1\n",
    "print(vocab_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Length of sequences are:  3889\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[ 17,  53],\n",
       "       [ 53, 293],\n",
       "       [293,   2],\n",
       "       [  2,  18],\n",
       "       [ 18, 729],\n",
       "       [729, 135],\n",
       "       [135, 730],\n",
       "       [730, 294],\n",
       "       [294,   8],\n",
       "       [  8, 731]])"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sequences = []\n",
    "\n",
    "for i in range(1, len(sequence_data)):\n",
    "    words = sequence_data[i-1:i+1]\n",
    "    sequences.append(words)\n",
    "    \n",
    "print(\"The Length of sequences are: \", len(sequences))\n",
    "sequences = np.array(sequences)\n",
    "sequences[:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = []\n",
    "y = []\n",
    "\n",
    "for i in sequences:\n",
    "    X.append(i[0])\n",
    "    y.append(i[1])\n",
    "    \n",
    "X = np.array(X)\n",
    "y = np.array(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Data is:  [ 17  53 293   2  18]\n",
      "The responses are:  [ 53 293   2  18 729]\n"
     ]
    }
   ],
   "source": [
    "print(\"The Data is: \", X[:5])\n",
    "print(\"The responses are: \", y[:5])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 0., 0., ..., 0., 0., 0.],\n",
       "       [0., 0., 0., ..., 0., 0., 0.],\n",
       "       [0., 0., 1., ..., 0., 0., 0.],\n",
       "       [0., 0., 0., ..., 0., 0., 0.],\n",
       "       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = to_categorical(y, num_classes=vocab_size)\n",
    "y[:5]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Creating the Model:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "model.add(Embedding(vocab_size, 10, input_length=1))\n",
    "model.add(LSTM(1000, return_sequences=True))\n",
    "model.add(LSTM(1000))\n",
    "model.add(Dense(1000, activation=\"relu\"))\n",
    "model.add(Dense(vocab_size, activation=\"softmax\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "embedding (Embedding)        (None, 1, 10)             26170     \n",
      "_________________________________________________________________\n",
      "lstm (LSTM)                  (None, 1, 1000)           4044000   \n",
      "_________________________________________________________________\n",
      "lstm_1 (LSTM)                (None, 1000)              8004000   \n",
      "_________________________________________________________________\n",
      "dense (Dense)                (None, 1000)              1001000   \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 2617)              2619617   \n",
      "=================================================================\n",
      "Total params: 15,694,787\n",
      "Trainable params: 15,694,787\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Plot The Model:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from tensorflow import keras\n",
    "from keras.utils.vis_utils import plot_model\n",
    "\n",
    "keras.utils.plot_model(model, to_file='model.png', show_layer_names=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Callbacks:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.callbacks import ModelCheckpoint\n",
    "from tensorflow.keras.callbacks import ReduceLROnPlateau\n",
    "from tensorflow.keras.callbacks import TensorBoard\n",
    "\n",
    "checkpoint = ModelCheckpoint(\"nextword1.h5\", monitor='loss', verbose=1,\n",
    "    save_best_only=True, mode='auto')\n",
    "\n",
    "reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)\n",
    "\n",
    "logdir='logsnextword1'\n",
    "tensorboard_Visualization = TensorBoard(log_dir=logdir)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Compile The Model:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=\"categorical_crossentropy\", optimizer=Adam(lr=0.001))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Fit The Model:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 3889 samples\n",
      "Epoch 1/150\n",
      "3712/3889 [===========================>..] - ETA: 0s - loss: 7.8752\n",
      "Epoch 00001: loss improved from inf to 7.87560, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 5s 1ms/sample - loss: 7.8756\n",
      "Epoch 2/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 7.8587\n",
      "Epoch 00002: loss improved from 7.87560 to 7.86009, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 331us/sample - loss: 7.8601\n",
      "Epoch 3/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 7.8187\n",
      "Epoch 00003: loss improved from 7.86009 to 7.81623, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 372us/sample - loss: 7.8162\n",
      "Epoch 4/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 7.6399\n",
      "Epoch 00004: loss improved from 7.81623 to 7.63961, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 327us/sample - loss: 7.6396\n",
      "Epoch 5/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 7.4280\n",
      "Epoch 00005: loss improved from 7.63961 to 7.42898, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 363us/sample - loss: 7.4290\n",
      "Epoch 6/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 7.2234\n",
      "Epoch 00006: loss improved from 7.42898 to 7.23395, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 335us/sample - loss: 7.2339\n",
      "Epoch 7/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 6.9898\n",
      "Epoch 00007: loss improved from 7.23395 to 6.99421, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 369us/sample - loss: 6.9942\n",
      "Epoch 8/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 6.7034\n",
      "Epoch 00008: loss improved from 6.99421 to 6.70877, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 335us/sample - loss: 6.7088\n",
      "Epoch 9/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 6.4315\n",
      "Epoch 00009: loss improved from 6.70877 to 6.44175, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 304us/sample - loss: 6.4417\n",
      "Epoch 10/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 6.1955\n",
      "Epoch 00010: loss improved from 6.44175 to 6.21069, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 329us/sample - loss: 6.2107\n",
      "Epoch 11/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 6.0073\n",
      "Epoch 00011: loss improved from 6.21069 to 6.01610, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 309us/sample - loss: 6.0161\n",
      "Epoch 12/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 5.8053\n",
      "Epoch 00012: loss improved from 6.01610 to 5.81236, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 346us/sample - loss: 5.8124\n",
      "Epoch 13/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 5.6305\n",
      "Epoch 00013: loss improved from 5.81236 to 5.64033, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 321us/sample - loss: 5.6403\n",
      "Epoch 14/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 5.4578\n",
      "Epoch 00014: loss improved from 5.64033 to 5.47233, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 312us/sample - loss: 5.4723\n",
      "Epoch 15/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 5.2988\n",
      "Epoch 00015: loss improved from 5.47233 to 5.30519, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 343us/sample - loss: 5.3052\n",
      "Epoch 16/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 5.1645\n",
      "Epoch 00016: loss improved from 5.30519 to 5.17411, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 326us/sample - loss: 5.1741\n",
      "Epoch 17/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 5.0464\n",
      "Epoch 00017: loss improved from 5.17411 to 5.07256, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 324us/sample - loss: 5.0726\n",
      "Epoch 18/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 4.9512\n",
      "Epoch 00018: loss improved from 5.07256 to 4.96286, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 332us/sample - loss: 4.9629\n",
      "Epoch 19/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 4.8412\n",
      "Epoch 00019: loss improved from 4.96286 to 4.85318, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 338us/sample - loss: 4.8532\n",
      "Epoch 20/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 4.7396\n",
      "Epoch 00020: loss improved from 4.85318 to 4.76163, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 362us/sample - loss: 4.7616\n",
      "Epoch 21/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 4.6530- ETA: 0s - loss: 4.60\n",
      "Epoch 00021: loss improved from 4.76163 to 4.67074, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 351us/sample - loss: 4.6707\n",
      "Epoch 22/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 4.5669\n",
      "Epoch 00022: loss improved from 4.67074 to 4.57855, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 351us/sample - loss: 4.5785\n",
      "Epoch 23/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 4.4609\n",
      "Epoch 00023: loss improved from 4.57855 to 4.47391, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 341us/sample - loss: 4.4739\n",
      "Epoch 24/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 4.3753\n",
      "Epoch 00024: loss improved from 4.47391 to 4.39767, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 337us/sample - loss: 4.3977\n",
      "Epoch 25/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 4.3182\n",
      "Epoch 00025: loss improved from 4.39767 to 4.33766, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 372us/sample - loss: 4.3377\n",
      "Epoch 26/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 4.2283\n",
      "Epoch 00026: loss improved from 4.33766 to 4.25059, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 349us/sample - loss: 4.2506\n",
      "Epoch 27/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 4.1376\n",
      "Epoch 00027: loss improved from 4.25059 to 4.15881, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 349us/sample - loss: 4.1588\n",
      "Epoch 28/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 4.0388\n",
      "Epoch 00028: loss improved from 4.15881 to 4.04326, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 354us/sample - loss: 4.0433\n",
      "Epoch 29/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 3.9224\n",
      "Epoch 00029: loss improved from 4.04326 to 3.93666, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 348us/sample - loss: 3.9367\n",
      "Epoch 30/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 3.7954\n",
      "Epoch 00030: loss improved from 3.93666 to 3.81460, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 330us/sample - loss: 3.8146\n",
      "Epoch 31/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 3.6849\n",
      "Epoch 00031: loss improved from 3.81460 to 3.70621, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 367us/sample - loss: 3.7062\n",
      "Epoch 32/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 3.5543\n",
      "Epoch 00032: loss improved from 3.70621 to 3.57647, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 338us/sample - loss: 3.5765\n",
      "Epoch 33/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 3.4483\n",
      "Epoch 00033: loss improved from 3.57647 to 3.45974, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 346us/sample - loss: 3.4597\n",
      "Epoch 34/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 3.3385\n",
      "Epoch 00034: loss improved from 3.45974 to 3.35306, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 2s 395us/sample - loss: 3.3531\n",
      "Epoch 35/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 3.2418\n",
      "Epoch 00035: loss improved from 3.35306 to 3.25955, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 321us/sample - loss: 3.2596\n",
      "Epoch 36/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 3.1403\n",
      "Epoch 00036: loss improved from 3.25955 to 3.15727, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 347us/sample - loss: 3.1573\n",
      "Epoch 37/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 3.0100\n",
      "Epoch 00037: loss improved from 3.15727 to 3.04076, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 366us/sample - loss: 3.0408\n",
      "Epoch 38/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.9453\n",
      "Epoch 00038: loss improved from 3.04076 to 2.96700, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 321us/sample - loss: 2.9670\n",
      "Epoch 39/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.8686\n",
      "Epoch 00039: loss improved from 2.96700 to 2.89164, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 369us/sample - loss: 2.8916\n",
      "Epoch 40/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.8006\n",
      "Epoch 00040: loss improved from 2.89164 to 2.82129, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 338us/sample - loss: 2.8213\n",
      "Epoch 41/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.7267\n",
      "Epoch 00041: loss improved from 2.82129 to 2.74699, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 352us/sample - loss: 2.7470\n",
      "Epoch 42/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.6175\n",
      "Epoch 00042: loss improved from 2.74699 to 2.64779, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 375us/sample - loss: 2.6478\n",
      "Epoch 43/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.5763\n",
      "Epoch 00043: loss improved from 2.64779 to 2.59631, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 318us/sample - loss: 2.5963\n",
      "Epoch 44/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.5125\n",
      "Epoch 00044: loss improved from 2.59631 to 2.53560, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 366us/sample - loss: 2.5356\n",
      "Epoch 45/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.4683\n",
      "Epoch 00045: loss improved from 2.53560 to 2.48991, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 364us/sample - loss: 2.4899\n",
      "Epoch 46/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.4373\n",
      "Epoch 00046: loss improved from 2.48991 to 2.45095, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 337us/sample - loss: 2.4509\n",
      "Epoch 47/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.3554\n",
      "Epoch 00047: loss improved from 2.45095 to 2.38200, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 340us/sample - loss: 2.3820\n",
      "Epoch 48/150\n",
      "3840/3889 [============================>.] - ETA: 0s - loss: 2.3300\n",
      "Epoch 00048: loss improved from 2.38200 to 2.33441, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 374us/sample - loss: 2.3344\n",
      "Epoch 49/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.2349\n",
      "Epoch 00049: loss improved from 2.33441 to 2.26422, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 347us/sample - loss: 2.2642\n",
      "Epoch 50/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.1906\n",
      "Epoch 00050: loss improved from 2.26422 to 2.21210, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 326us/sample - loss: 2.2121\n",
      "Epoch 51/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.1362\n",
      "Epoch 00051: loss improved from 2.21210 to 2.16222, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 363us/sample - loss: 2.1622\n",
      "Epoch 52/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.0559\n",
      "Epoch 00052: loss improved from 2.16222 to 2.07998, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 346us/sample - loss: 2.0800\n",
      "Epoch 53/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 2.0071\n",
      "Epoch 00053: loss improved from 2.07998 to 2.03807, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 343us/sample - loss: 2.0381\n",
      "Epoch 54/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.9955\n",
      "Epoch 00054: loss improved from 2.03807 to 2.01186, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 363us/sample - loss: 2.0119\n",
      "Epoch 55/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.9206\n",
      "Epoch 00055: loss improved from 2.01186 to 1.94076, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 335us/sample - loss: 1.9408\n",
      "Epoch 56/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.8732\n",
      "Epoch 00056: loss improved from 1.94076 to 1.89430, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 355us/sample - loss: 1.8943\n",
      "Epoch 57/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.8295\n",
      "Epoch 00057: loss improved from 1.89430 to 1.84639, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 341us/sample - loss: 1.8464\n",
      "Epoch 58/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.8001\n",
      "Epoch 00058: loss improved from 1.84639 to 1.81274, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 369us/sample - loss: 1.8127\n",
      "Epoch 59/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.7316\n",
      "Epoch 00059: loss improved from 1.81274 to 1.75823, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 352us/sample - loss: 1.7582\n",
      "Epoch 60/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.7274\n",
      "Epoch 00060: loss improved from 1.75823 to 1.74511, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 334us/sample - loss: 1.7451\n",
      "Epoch 61/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.6874\n",
      "Epoch 00061: loss improved from 1.74511 to 1.69725, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 335us/sample - loss: 1.6973\n",
      "Epoch 62/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.6354\n",
      "Epoch 00062: loss improved from 1.69725 to 1.65130, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 323us/sample - loss: 1.6513\n",
      "Epoch 63/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.5929\n",
      "Epoch 00063: loss improved from 1.65130 to 1.61970, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 324us/sample - loss: 1.6197\n",
      "Epoch 64/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.5776\n",
      "Epoch 00064: loss improved from 1.61970 to 1.59771, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 324us/sample - loss: 1.5977\n",
      "Epoch 65/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.5653\n",
      "Epoch 00065: loss improved from 1.59771 to 1.58945, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 325us/sample - loss: 1.5895\n",
      "Epoch 66/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.5389\n",
      "Epoch 00066: loss improved from 1.58945 to 1.55059, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 380us/sample - loss: 1.5506\n",
      "Epoch 67/150\n",
      "3776/3889 [============================>.] - ETA: 0s - loss: 1.4600\n",
      "Epoch 00067: loss improved from 1.55059 to 1.47310, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 360us/sample - loss: 1.4731\n",
      "Epoch 68/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.3935\n",
      "Epoch 00068: loss improved from 1.47310 to 1.41295, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 348us/sample - loss: 1.4130\n",
      "Epoch 69/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.3613\n",
      "Epoch 00069: loss improved from 1.41295 to 1.37151, saving model to nextword1.h5\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3889/3889 [==============================] - 1s 370us/sample - loss: 1.3715\n",
      "Epoch 70/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.3227\n",
      "Epoch 00070: loss improved from 1.37151 to 1.33558, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 358us/sample - loss: 1.3356\n",
      "Epoch 71/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.2890\n",
      "Epoch 00071: loss improved from 1.33558 to 1.31252, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 319us/sample - loss: 1.3125\n",
      "Epoch 72/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.2673\n",
      "Epoch 00072: loss improved from 1.31252 to 1.28071, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 358us/sample - loss: 1.2807\n",
      "Epoch 73/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.2503\n",
      "Epoch 00073: loss improved from 1.28071 to 1.26773, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 2s 394us/sample - loss: 1.2677\n",
      "Epoch 74/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.2162\n",
      "Epoch 00074: loss improved from 1.26773 to 1.23261, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 313us/sample - loss: 1.2326\n",
      "Epoch 75/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.1998\n",
      "Epoch 00075: loss improved from 1.23261 to 1.22247, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 357us/sample - loss: 1.2225\n",
      "Epoch 76/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.1733\n",
      "Epoch 00076: loss improved from 1.22247 to 1.20326, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 341us/sample - loss: 1.2033\n",
      "Epoch 77/150\n",
      "3776/3889 [============================>.] - ETA: 0s - loss: 1.1803\n",
      "Epoch 00077: loss improved from 1.20326 to 1.18947, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 343us/sample - loss: 1.1895\n",
      "Epoch 78/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.1621\n",
      "Epoch 00078: loss improved from 1.18947 to 1.17934, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 343us/sample - loss: 1.1793\n",
      "Epoch 79/150\n",
      "3776/3889 [============================>.] - ETA: 0s - loss: 1.1849\n",
      "Epoch 00079: loss did not improve from 1.17934\n",
      "3889/3889 [==============================] - 1s 244us/sample - loss: 1.1901\n",
      "Epoch 80/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.1713\n",
      "Epoch 00080: loss improved from 1.17934 to 1.17593, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 384us/sample - loss: 1.1759\n",
      "Epoch 81/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.1598\n",
      "Epoch 00081: loss did not improve from 1.17593\n",
      "3889/3889 [==============================] - 1s 255us/sample - loss: 1.1792\n",
      "Epoch 82/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.1384\n",
      "Epoch 00082: loss improved from 1.17593 to 1.14731, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 305us/sample - loss: 1.1473\n",
      "Epoch 83/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.1196\n",
      "Epoch 00083: loss improved from 1.14731 to 1.13632, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 378us/sample - loss: 1.1363\n",
      "Epoch 84/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.1390\n",
      "Epoch 00084: loss did not improve from 1.13632\n",
      "3889/3889 [==============================] - 1s 324us/sample - loss: 1.1542\n",
      "Epoch 85/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.1067\n",
      "Epoch 00085: loss improved from 1.13632 to 1.13034, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 317us/sample - loss: 1.1303\n",
      "Epoch 86/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.0991\n",
      "Epoch 00086: loss improved from 1.13034 to 1.11426, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 369us/sample - loss: 1.1143\n",
      "Epoch 87/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.0856\n",
      "Epoch 00087: loss improved from 1.11426 to 1.10867, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 378us/sample - loss: 1.1087\n",
      "Epoch 88/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.0854\n",
      "Epoch 00088: loss improved from 1.10867 to 1.09807, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 2s 393us/sample - loss: 1.0981\n",
      "Epoch 89/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.0814\n",
      "Epoch 00089: loss improved from 1.09807 to 1.09314, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 371us/sample - loss: 1.0931\n",
      "Epoch 90/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.0725\n",
      "Epoch 00090: loss improved from 1.09314 to 1.08705, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 2s 394us/sample - loss: 1.0870\n",
      "Epoch 91/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.0691\n",
      "Epoch 00091: loss improved from 1.08705 to 1.08696, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 370us/sample - loss: 1.0870\n",
      "Epoch 92/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.0672\n",
      "Epoch 00092: loss improved from 1.08696 to 1.07776, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 369us/sample - loss: 1.0778\n",
      "Epoch 93/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.0441\n",
      "Epoch 00093: loss improved from 1.07776 to 1.05813, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 2s 390us/sample - loss: 1.0581\n",
      "Epoch 94/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.0451\n",
      "Epoch 00094: loss did not improve from 1.05813\n",
      "3889/3889 [==============================] - 1s 241us/sample - loss: 1.0633\n",
      "Epoch 95/150\n",
      "3840/3889 [============================>.] - ETA: 0s - loss: 1.0503\n",
      "Epoch 00095: loss improved from 1.05813 to 1.05076, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 2s 394us/sample - loss: 1.0508\n",
      "Epoch 96/150\n",
      "3840/3889 [============================>.] - ETA: 0s - loss: 1.0671\n",
      "Epoch 00096: loss did not improve from 1.05076\n",
      "3889/3889 [==============================] - 1s 327us/sample - loss: 1.0668\n",
      "Epoch 97/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.0559\n",
      "Epoch 00097: loss did not improve from 1.05076\n",
      "3889/3889 [==============================] - 1s 255us/sample - loss: 1.0644\n",
      "Epoch 98/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 1.0557\n",
      "Epoch 00098: loss did not improve from 1.05076\n",
      "\n",
      "Epoch 00098: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.\n",
      "3889/3889 [==============================] - 1s 279us/sample - loss: 1.0708\n",
      "Epoch 99/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.8185\n",
      "Epoch 00099: loss improved from 1.05076 to 0.81769, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 363us/sample - loss: 0.8177\n",
      "Epoch 100/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.7314\n",
      "Epoch 00100: loss improved from 0.81769 to 0.73557, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 362us/sample - loss: 0.7356\n",
      "Epoch 101/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.7066\n",
      "Epoch 00101: loss improved from 0.73557 to 0.70975, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 374us/sample - loss: 0.7097\n",
      "Epoch 102/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6923\n",
      "Epoch 00102: loss improved from 0.70975 to 0.69729, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 377us/sample - loss: 0.6973\n",
      "Epoch 103/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6857\n",
      "Epoch 00103: loss improved from 0.69729 to 0.69078, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 374us/sample - loss: 0.6908\n",
      "Epoch 104/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6814\n",
      "Epoch 00104: loss improved from 0.69078 to 0.68539, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 372us/sample - loss: 0.6854\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 105/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6752- ETA: 0s - los\n",
      "Epoch 00105: loss improved from 0.68539 to 0.68348, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 2s 411us/sample - loss: 0.6835\n",
      "Epoch 106/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6793\n",
      "Epoch 00106: loss improved from 0.68348 to 0.68141, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 381us/sample - loss: 0.6814\n",
      "Epoch 107/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6729\n",
      "Epoch 00107: loss improved from 0.68141 to 0.67983, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 2s 413us/sample - loss: 0.6798\n",
      "Epoch 108/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6734\n",
      "Epoch 00108: loss did not improve from 0.67983\n",
      "3889/3889 [==============================] - 1s 297us/sample - loss: 0.6801\n",
      "Epoch 109/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6769\n",
      "Epoch 00109: loss improved from 0.67983 to 0.67935, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 314us/sample - loss: 0.6794\n",
      "Epoch 110/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6763\n",
      "Epoch 00110: loss improved from 0.67935 to 0.67788, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 349us/sample - loss: 0.6779\n",
      "Epoch 111/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6808\n",
      "Epoch 00111: loss did not improve from 0.67788\n",
      "3889/3889 [==============================] - 1s 292us/sample - loss: 0.6784\n",
      "Epoch 112/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6728\n",
      "Epoch 00112: loss did not improve from 0.67788\n",
      "3889/3889 [==============================] - 1s 241us/sample - loss: 0.6782\n",
      "Epoch 113/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6707\n",
      "Epoch 00113: loss improved from 0.67788 to 0.67655, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 312us/sample - loss: 0.6766\n",
      "Epoch 114/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6726\n",
      "Epoch 00114: loss did not improve from 0.67655\n",
      "3889/3889 [==============================] - 1s 236us/sample - loss: 0.6768\n",
      "Epoch 115/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6741\n",
      "Epoch 00115: loss improved from 0.67655 to 0.67492, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 309us/sample - loss: 0.6749\n",
      "Epoch 116/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6697\n",
      "Epoch 00116: loss did not improve from 0.67492\n",
      "3889/3889 [==============================] - 1s 239us/sample - loss: 0.6767\n",
      "Epoch 117/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6695\n",
      "Epoch 00117: loss did not improve from 0.67492\n",
      "3889/3889 [==============================] - 1s 239us/sample - loss: 0.6762\n",
      "Epoch 118/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6687\n",
      "Epoch 00118: loss did not improve from 0.67492\n",
      "\n",
      "Epoch 00118: ReduceLROnPlateau reducing learning rate to 0.0001.\n",
      "3889/3889 [==============================] - 1s 236us/sample - loss: 0.6754\n",
      "Epoch 119/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6331\n",
      "Epoch 00119: loss improved from 0.67492 to 0.63376, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 312us/sample - loss: 0.6338\n",
      "Epoch 120/150\n",
      "3840/3889 [============================>.] - ETA: 0s - loss: 0.6335\n",
      "Epoch 00120: loss improved from 0.63376 to 0.63225, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 2s 389us/sample - loss: 0.6322\n",
      "Epoch 121/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6284\n",
      "Epoch 00121: loss improved from 0.63225 to 0.63113, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 366us/sample - loss: 0.6311\n",
      "Epoch 122/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6256\n",
      "Epoch 00122: loss did not improve from 0.63113\n",
      "3889/3889 [==============================] - 1s 270us/sample - loss: 0.6321\n",
      "Epoch 123/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6317\n",
      "Epoch 00123: loss did not improve from 0.63113\n",
      "3889/3889 [==============================] - 1s 240us/sample - loss: 0.6312\n",
      "Epoch 124/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6226\n",
      "Epoch 00124: loss improved from 0.63113 to 0.63077, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 307us/sample - loss: 0.6308\n",
      "Epoch 125/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6297\n",
      "Epoch 00125: loss did not improve from 0.63077\n",
      "3889/3889 [==============================] - 1s 239us/sample - loss: 0.6312\n",
      "Epoch 126/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6310\n",
      "Epoch 00126: loss improved from 0.63077 to 0.62999, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 2s 426us/sample - loss: 0.6300\n",
      "Epoch 127/150\n",
      "3840/3889 [============================>.] - ETA: 0s - loss: 0.6318\n",
      "Epoch 00127: loss did not improve from 0.62999\n",
      "3889/3889 [==============================] - 1s 255us/sample - loss: 0.6301\n",
      "Epoch 128/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6250\n",
      "Epoch 00128: loss did not improve from 0.62999\n",
      "3889/3889 [==============================] - 1s 270us/sample - loss: 0.6301\n",
      "Epoch 129/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6258\n",
      "Epoch 00129: loss did not improve from 0.62999\n",
      "3889/3889 [==============================] - 1s 252us/sample - loss: 0.6302\n",
      "Epoch 130/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6189\n",
      "Epoch 00130: loss improved from 0.62999 to 0.62929, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 321us/sample - loss: 0.6293\n",
      "Epoch 131/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6209\n",
      "Epoch 00131: loss improved from 0.62929 to 0.62818, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 375us/sample - loss: 0.6282\n",
      "Epoch 132/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6254\n",
      "Epoch 00132: loss did not improve from 0.62818\n",
      "3889/3889 [==============================] - 1s 327us/sample - loss: 0.6285\n",
      "Epoch 133/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6216\n",
      "Epoch 00133: loss did not improve from 0.62818\n",
      "3889/3889 [==============================] - 1s 252us/sample - loss: 0.6298\n",
      "Epoch 134/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6237\n",
      "Epoch 00134: loss did not improve from 0.62818\n",
      "3889/3889 [==============================] - 1s 238us/sample - loss: 0.6291\n",
      "Epoch 135/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6272\n",
      "Epoch 00135: loss did not improve from 0.62818\n",
      "3889/3889 [==============================] - 1s 285us/sample - loss: 0.6283\n",
      "Epoch 136/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6247\n",
      "Epoch 00136: loss improved from 0.62818 to 0.62810, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 317us/sample - loss: 0.6281\n",
      "Epoch 137/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6254- ETA: 0s - loss: 0.62\n",
      "Epoch 00137: loss did not improve from 0.62810\n",
      "3889/3889 [==============================] - 1s 268us/sample - loss: 0.6291\n",
      "Epoch 138/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6234\n",
      "Epoch 00138: loss improved from 0.62810 to 0.62745, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 331us/sample - loss: 0.6274\n",
      "Epoch 139/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6240\n",
      "Epoch 00139: loss did not improve from 0.62745\n",
      "3889/3889 [==============================] - 1s 235us/sample - loss: 0.6282\n",
      "Epoch 140/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6221\n",
      "Epoch 00140: loss did not improve from 0.62745\n",
      "3889/3889 [==============================] - 1s 274us/sample - loss: 0.6279\n",
      "Epoch 141/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6226\n",
      "Epoch 00141: loss improved from 0.62745 to 0.62663, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 313us/sample - loss: 0.6266\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 142/150\n",
      "3776/3889 [============================>.] - ETA: 0s - loss: 0.6279\n",
      "Epoch 00142: loss did not improve from 0.62663\n",
      "3889/3889 [==============================] - 1s 319us/sample - loss: 0.6278\n",
      "Epoch 143/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6270\n",
      "Epoch 00143: loss did not improve from 0.62663\n",
      "3889/3889 [==============================] - 1s 249us/sample - loss: 0.6268\n",
      "Epoch 144/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6195\n",
      "Epoch 00144: loss did not improve from 0.62663\n",
      "3889/3889 [==============================] - 1s 239us/sample - loss: 0.6276\n",
      "Epoch 145/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6204\n",
      "Epoch 00145: loss did not improve from 0.62663\n",
      "3889/3889 [==============================] - 1s 239us/sample - loss: 0.6271\n",
      "Epoch 146/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6203\n",
      "Epoch 00146: loss did not improve from 0.62663\n",
      "3889/3889 [==============================] - 1s 239us/sample - loss: 0.6267\n",
      "Epoch 147/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6190\n",
      "Epoch 00147: loss improved from 0.62663 to 0.62624, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 2s 403us/sample - loss: 0.6262\n",
      "Epoch 148/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6177\n",
      "Epoch 00148: loss improved from 0.62624 to 0.62619, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 375us/sample - loss: 0.6262\n",
      "Epoch 149/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6208\n",
      "Epoch 00149: loss did not improve from 0.62619\n",
      "3889/3889 [==============================] - 1s 290us/sample - loss: 0.6268\n",
      "Epoch 150/150\n",
      "3648/3889 [===========================>..] - ETA: 0s - loss: 0.6263\n",
      "Epoch 00150: loss improved from 0.62619 to 0.62493, saving model to nextword1.h5\n",
      "3889/3889 [==============================] - 1s 369us/sample - loss: 0.6249\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<tensorflow.python.keras.callbacks.History at 0x20eef7fa308>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X, y, epochs=150, batch_size=64, callbacks=[checkpoint, reduce, tensorboard_Visualization])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Graph:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABZ0AAAKhCAYAAAA7VgBrAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAOVWSURBVHhe7P19dFVlnqAN88eznj9mrWfmXfPH45pZzzxTmVnraWZcbb32Wnb1O13VzsT3raKpstRyWhurKjV2q1WlVbYWDSpIAZaaRgUUKRQ7iBSCNiiCH4glGhHCZ/iQABLASCRAiCYhIJ8Bfu/+7b3vc/bZ2efk5JDk7F9yXWtdi2Tf++yzzz4hJhe39x5x6uERgoiIiIiIiIiIiIjYHxKdEREREREREREREbHfJDojIiIiIiIiIiIiYr9JdEZERERERERERETEfpPojIiIiIiIiIiIiIj9JtEZEREREREREREREftNojMiIiIiIiIiIiIi9ptEZ0RERERERERERETsN4nOiIiIiIiIiIiIiNhvEp0RERERERERERERsd8kOiMiIiIiIiIiIiJiv0l0RkRERERERERERMR+k+iMiIiIiIiIiIiIiP0m0RkRERERERERERER+02iMyIiIiIiIiIiIiL2m0RnREREREREREREROw3ic6IiIiIiIiIiIiI2G8SnRERERERERERERGx3yQ6IyIiIiIiIiIiImK/SXRGRERERERERERExH6T6IyIiIiIiIiIiIiI/SbRGREREREREREREbGA51beJ3F0W9K+g+251/+XdG+bLxePbA/PTPyPddvZRTckPmagJTojIiIiIiIiIiIiFvDCnuVhzs3Sve2lxH0Hy7OLbpRLnZ+HZ5Mf3Uf3TTrGQDlo0fn0vB/JudduK86Xr048hiXPvJzwulL/2sbKhTPhF2PT2IRxRERERERERETE/vX0U9+Q7rqZPWfqett0LLpvuULvxc8/Cs8si25L2ncw7F7/dHgWGpUP+rOuz/z+/50Z1491m4459HpGjzGQDlp0Pt/UGb68ImirSTyGJbvbwteSxMVOudj4lJxNeFx5JTojIiIiIiIiDhs/WCoXD9fLhQ+Cz8/vqg8+r/3LnP3Oba71ttdK95LstsAbpHu/PuY96X45PuZZc590f7reP6bvofVyYfN9ckbHljwnF9z2JHdN8o7xE+lu8j5uek7OPXyNnN+jYwnn8YdngmPtf0bOPjwp73EvbP5J7uMwFUbjaT5c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eOafv+2bNJZap/4r/+vC3HkjIiIiIiIiDrJujWJFZw+7+KzB9kzNtf42hy5rGX/8QFhoiY1iGYilNZw6Q9mRFOsd8e3R2eIDPctZTW10Vs+8vUguulDsuNgpF7ffE/4vGX2Jzp7P3SPdrdHaGtL5Qc5san+G8e492SDtiO1XcnQ+tUcubLond9F8z9NLZsvFk7nx173e6L49b4h4xnv9r2evVVuNv3+h8zs1w7t2nbHQ3DJbLoQzronOiIiIiIiIiIg40LoZz4UY6BnOcTVwqxqPNYAXWrtZx3Qf///ADh+XdMz+NHrNzr3zD4n7RHUznJXBupbli8598bnRcu612+Tcy1cnj/fZq+WsHu+1H8lZf3ZzPivk7KvF7NfPFvl6z7x8+ed2et6P/Oc6+1zyOCIiIiIiIiIi4kCq6zRrDI3OfNYlIHTbQK7h3Bc1JscZjMCcz2h41jWeNT5HZz7rx7otuvb0YAVn1UZ0RkRERERERERERCyTF/YsD9NtFp3lnLTvYKnLZLg1nguh+wzGkhpRic6IiIiIiIiIiIiIBYwuUeHIdzO/wVaDsgbw6Exx/Vi3DXZsdhKdEREREREREREREbHfJDojIiIiIiIiIiIiYr9JdEZEREREREREHGZePFyP2G8mfY3h8JbojIiIiIiIiIiIiIj9JtEZEREREREREREREftNojMiIiIiIiIiIiIi9ptEZ0RERERERERERETsN4nOiIiIiIiIiIiIiNhvEp0RERERERERERERsd8kOiMiIiIiIiIiIiJiv0l0RkRERERERERERMR+k+iMiIiIiIiIiIiIiP0m0RkRERERERERERER+02iMyIiIiIiIiIiIiL2m0RnREREREREREREROw3ic6IiIiIiIiIiIiI2G8SnRERERERERERERGx3yQ6IyIiIiIiIiIiImK/SXRGRERERERERERExH6T6IyIiIiIiIiIiIiI/SbRGYet0P+cXTIm8VojIiIiIiIiIuLwkeiMw1bof4jOiIiIiIiIiIg4YtmyZfLGG2/I8uXLZcWKFYiIiIiIiIiIiIiIJTti1apV8sc//lFWr14tH3zwASIiIiIiIiIiIiJiyY5Yt26dbNiwQTZt2iSbN2+WLVu2ICIiIiIiIiIiIiL2WW3MI7Zv3y47d+6UXbt2yZ49e3w//fRTRERERERERERERMSidX15xN69e+XAgQPy+eefy8GDB6W5uVm++OILRERERERERERERMSi1basjXmExuaWlhY5evSotLa2yrFjxxARERERERERERER+6w25hGnTp0SRERERERERERERMT+kOiMiIiIiIiIiIiIiP0m0RkRERERERERERER+02iMyIiIiIiIiIiIiL2m0RnREREREREREREROw3ic6IiIiIiIiIiIiI2G8SnRERERERERERERGx3yQ6IyIiIiIiIiIiImK/SXRGRERERERERERExH6T6IyIiIiIiIiIiIiI/SbRGRERERERERERERH7TaIzIiIiIiIiIiIiIvabRGdERERERERERERE7DeJzoiIiIiIiIiIiIjYbxKdEREREREREREREbHfJDojIhpz1apV8pOf/ET+zb/5NzJixAj/T/1ctyftj4iIiIiIiIg4mBKdEREN+eMf/9gPzfnU+Jz0OOw/v/rqK3n33Xelo6MjcRwRL98DBw7IRx99JCdPnkwcR0RERETEdEt0RkQ0YjQ4T5gwITOzWf/Uz0sJz59++qls3Lgx0SNHjsiJEyfk448/lv379/v7Hz16VB555BF55513ehxrONje3i6zZ8+WN954w782SfsMlsP9vSjV1tZWef/99/1/PEgax/KpX8vPPPOM/w86+j7px7rt66+/TtwfERERERHTK9EZEdGAGpZdVN6wYUPiPrrd7VPsUhsvvfSSPPDAAzJ16lQ/YEbdsWOHH34effRReeWVV/z946Hz+PHjsnLlStm6dWvOcftLPa4eX58naXww1fC1bNkyee6556SzszNxn8GU6Fya+jWlX/P69Z00jn2zP78HRKOzft7Y2Oh/jes/jsX3RURERETEdFu26Lxu3TqZPHmyjBs3riT1sXqMpGMjIg413SxnndGcNO50M551/6TxuBqdo5EnSQ2s7n9xj4dOfZybjRh9TH8Zj1Dl9LPPPvNf+/bt2xPHB1uic2nqPx6k4etpqNif3wPif9/1vVq8eLHMmzev7P9nASIiIiIi9s2yRefLCc5OPUbSsRERh5r/+l//64KznJ1uRrTeXDBpPG5v0VmDs85gbGpq8j+Phk7dtnbtWnn88cflxRdflE2bNuUsWaD7rlixQhYtWuQv1xGdHayP3blzpx9yly5d2mOWpB5Hj6fH1ePr87hzUFtaWvxjL1y4UN57770e56/P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P46Rz47WwscuZby3x/Swrff9+3zMfrbge9DVIa2tvcyuK+I1Dpp6voeZDYiIiIiIiIiIiEPD4XsjQURERERERERERETsd8sWnSdPnpwJx6Wqx0g6NiIiIiIiIiIiIiKWx7JF53Xr1l1WeNbH6jGSjo2IiIiIiIiIiIiI5bFs0RkREQs7duxYREREREREREQzuqZBdEZERERERERERETEfpPojIiIiIiIiIiIiIj9JtEZEREREREREREREftNojMiIiIiIiIiIiIi9ptEZ0RERERERERERETsN4nOiIiIiIiIiIiIiNhvEp0RERERERERERERsd8kOiMiIiIiIiIiIiJiv0l0RkRERERERERERMR+s2zRed26dTJ58mQZN25cSepj9RhJx0ZERERERERERETE8li26Hw5wdmpx0g6NiIiIiIiIiIiIiKWx7JF56SIXIpJx8bCdhxulY6u5DHrDtXX1tHaKq0dyWOIiIiIiIiIiIhp0lx0/t3vfifTp0/PfJ507LiNK6ql+sU6aU0YG37WyZSKCpmyNvi8eek9MvrepdLcY7+BsWP9TLn1tplSNyABNfe1DR2bZWFVhVQtag4+76iTmbfdKjPXd8T2Q0RERERERERELL+morMG55aWFuns7MyE56Rjx617pEIqqhYOWlhNk3WzR8vo2XWRbblhtnXVFLnzkVWDF+R31cidd9RIw4DMRh646NzzOg6msejc1SA1d9wpNbvi++FQ9euvv0ZERERERERETK3xlmEmOrvgrDQ3N2fWhE46dtxhHZ31tT+SPzoPLQcwOve4joNpLDrjsDD6jfvkyZOIiIiIiIiIiKk12jG0a6QmOhe6sWC+4KwmHTtuj+jcVi9zbr9GRlZ42yuukltnL5U5VdlY2byoytt/jqx68R655krdp0Kuum2O1Ldlj3mqrUEWPniTXOUfwxu/cYKs2Jcd95/z/pky57arvPEqWdgUbG+unSlVfzHSf0zF1TfJlFUFQmJXs9ROr8qcw8i/qJI5m7NLKgTnmRvTM3G0aaFUhecW6M4hN8z2OEbsOa+6cYqsag7H/GPeKlOeGSujddxF2OZVMuVGfZ36mJFyze2xaxV17RRvnylS538ensvSyOOvHC1jlzX2fJyzq1GWxq770k/deBHH62qVOu999c9fx733YEJkvMf7tjnfdYzpXbdVj2TPK/5e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\n",
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# https://stackoverflow.com/questions/26649716/how-to-show-pil-image-in-ipython-notebook\n",
    "# tensorboard --logdir=\"./logsnextword1\"\n",
    "# http://DESKTOP-U3TSCVT:6006/\n",
    "\n",
    "from IPython.display import Image \n",
    "pil_img = Image(filename='graph1.png')\n",
    "display(pil_img)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Observation:\n",
    "### We are able to develop a decent next word prediction model and are able to get a declining loss and an overall decent performance."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
