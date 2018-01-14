#! /usr/bin/env python3

import gc
import numpy as np
from random import shuffle

from keras import backend as K
from keras.utils import to_categorical
from keras.models import load_model

# data set
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
classes = len(alphabet)

# letter<->integer encoding
char_to_int = {c: i for i, c in enumerate(alphabet)}
int_to_char = {i: c for c, i in char_to_int.items()}

# encode inputs
X = [char_to_int[c] for c in alphabet[:25]]
# reshape to (samples, time steps, features)
X = np.reshape(X, (len(X), 1, 1))
# normalize
X = X / classes

# encode targets
y = [char_to_int[c] for c in alphabet[1:]]
# one hot encoding
y = to_categorical(y)

# model loading
model = load_model('keras.h5')

# model evaluation
scores = model.evaluate(X, y, verbose=0)
print('Model accuracy: {:.2f}%\n'.format(scores[1] * 100))

# finally, run some predictions (out of order)
l = list(alphabet[:25])
shuffle(l)
for a in l:
    # prepare input data
    x = np.reshape([char_to_int[a]], (1, 1, 1)) / classes
    # feed prepared input data to the model
    pred_y = model.predict(x, verbose=0)
    # convert output back to something lisible
    real_y = int_to_char[np.argmax(pred_y)]
    # check if output match our expectations
    expected = chr(ord(a) + 1)
    print('{} -> {} {}'.format(a, real_y, '' if real_y == expected else '*'))


# https://github.com/tensorflow/tensorflow/issues/3388#issuecomment-271107725
K.clear_session()
# https://github.com/tensorflow/tensorflow/issues/3388#issuecomment-268502675
gc.collect()
