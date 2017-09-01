#! /usr/bin/env python3

import gc

import numpy as np
np.random.seed(42) # noqa

import tensorflow as tf
tf.set_random_seed(42) # noqa

from random import shuffle
from shutil import rmtree
from math import floor

from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import GRU, ELU, Dropout, Dense
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.regularizers import l2

# data set
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# letter<->integer encoding
char_to_int = {c: i for i, c in enumerate(alphabet)}
int_to_char = {i: c for c, i in char_to_int.items()}

# encode inputs
X = [char_to_int[c] for c in alphabet[:25]]
# reshape to (samples, time steps, features)
X = np.reshape(X, (len(X), 1, 1))
# normalize
X = X / len(alphabet)

# encode targets
y = [char_to_int[c] for c in alphabet[1:]]
# one hot encoding
y = to_categorical(y)

# parameters
epochs = 500
l2_reg = 1e-4
dropout_prob = 0.1
min_lr = 0.001  # default Adam learning rate
max_lr = 0.01   # x10 = 96%
step_lr = epochs / 10
scale_lr = True


# tensorboard
tflogs = '/tmp/tflogs/'
rmtree(tflogs, ignore_errors=True)


class TensorBoardWithLR(TensorBoard):

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        super().on_epoch_end(epoch, logs)


# model definition
model = Sequential()

model.add(GRU(16, input_shape=(X.shape[1], X.shape[2]),
              implementation=2, return_sequences=True,
              kernel_regularizer=l2(l2_reg)))
model.add(ELU())
model.add(Dropout(dropout_prob))

model.add(GRU(16, input_shape=(X.shape[1], X.shape[2]),
              implementation=2, return_sequences=True,
              kernel_regularizer=l2(l2_reg)))
model.add(ELU())
model.add(Dropout(dropout_prob))

model.add(GRU(16, input_shape=(X.shape[1], X.shape[2]),
              implementation=2, return_sequences=False,
              kernel_regularizer=l2(l2_reg)))
model.add(ELU())
model.add(Dropout(dropout_prob))

model.add(Dense(y.shape[1], activation='softmax',
                kernel_regularizer=l2(l2_reg)))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


def learning_rate_scheduler(epoch):
    # https://arxiv.org/abs/1506.01186
    lr_cycle = floor(1 + epoch / (2 * step_lr))
    lr_x = abs(epoch / step_lr - 2 * lr_cycle + 1)
    lr_scale = 1 / (2 ** (lr_cycle - 1)) if scale_lr else 1
    return min_lr + (max_lr - min_lr) * max(0, 1 - lr_x) * lr_scale


# model training
model.fit(X, y,
          epochs=epochs,
          batch_size=1,
          shuffle=False,
          callbacks=[
              LearningRateScheduler(learning_rate_scheduler),
              TensorBoardWithLR(log_dir=tflogs),
          ],
          verbose=2)

# model saving
model.save('keras.h5')

# model summary
model.summary()

# model evaluation
scores = model.evaluate(X, y, verbose=0)
print('Model accuracy: {:.2f}%\n'.format(scores[1] * 100))

# finally, run some predictions (out of order)
l = list(alphabet[:25])
shuffle(l)
for a in l:
    # prepare input data
    x = np.reshape([char_to_int[a]], (1, 1, 1)) / len(alphabet)
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

print('tensorboard --logdir=/tmp/tflogs/')
