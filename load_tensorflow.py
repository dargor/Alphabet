#! /usr/bin/env python3

import numpy as np
import tensorflow as tf

from functools import reduce
from random import shuffle


def model_parameters():
    def variable_parameters(v):
        return reduce(lambda x, y: x * y, v.get_shape().as_list(), 1)
    return sum(variable_parameters(v) for v in tf.trainable_variables())


alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
classes = len(alphabet)

char_to_int = {c: i for i, c in enumerate(alphabet)}
int_to_char = {i: c for c, i in char_to_int.items()}

x_train = [char_to_int[c] for c in alphabet[:25]]
x_train = np.reshape(x_train, [-1, 1, 1])
x_train = x_train / classes

y_train = [char_to_int[c] for c in alphabet[1:]]
y_train = np.identity(classes)[y_train]

batch_size = 1
assert x_train.shape[0] % batch_size == 0
batches = x_train.shape[0] / batch_size


latest_checkpoint = tf.train.latest_checkpoint('.')
saver = tf.train.import_meta_graph('{}.meta'.format(latest_checkpoint))
with tf.Session() as sess:
    saver.restore(sess, latest_checkpoint)
    loss, accuracy, y = tf.get_collection('evaluations')
    yhat, x = tf.get_collection('predictions')

    # evaluation
    avg_loss = 0
    avg_accuracy = 0
    for n in range(x_train.shape[0]):
        l, a = sess.run([loss, accuracy], {
            x: [x_train[n]],
            y: [y_train[n]],
        })
        avg_loss += l
        avg_accuracy += a
    avg_loss /= batches
    avg_accuracy /= batches
    print('Model loss: {:.9f}'.format(avg_loss))
    print('Model accuracy: {:.9f}'.format(avg_accuracy))
    print('Model parameters: {}'.format(model_parameters()))

    # predictions
    l = list(alphabet[:25])
    shuffle(l)
    for a in l:
        # prepare input data
        _x = np.reshape([char_to_int[a]], [1, 1]) / classes
        # feed prepared input data to the model
        pred_y = yhat.eval({
            x: [_x],
        })
        # convert output back to something lisible
        real_y = int_to_char[np.argmax(pred_y)]
        # check if output match our expectations
        expected = chr(ord(a) + 1)
        print('{} -> {} {}'
              .format(a, real_y, '' if real_y == expected else '*'))
