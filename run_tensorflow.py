#! /usr/bin/env python3

import random as rn
rn.seed(42) # noqa

import numpy as np
np.random.seed(42) # noqa

import tensorflow as tf
tf.set_random_seed(42) # noqa

from functools import reduce
from random import shuffle
from shutil import rmtree
from math import floor


tflogs = '/tmp/tflogs/'
rmtree(tflogs, ignore_errors=True)


zeros = tf.zeros_initializer()


def model_parameters():
    def variable_parameters(v):
        return reduce(lambda x, y: x * y, v.get_shape().as_list(), 1)
    return sum(variable_parameters(v) for v in tf.trainable_variables())


def relu(x, name='relu'):
    with tf.variable_scope(name):
        zeros = tf.constant(0, tf.float32, x.shape)
        return tf.where(x >= 0, x, zeros)


def lrelu(x, alpha=0.2, name='lrelu'):
    with tf.variable_scope(name):
        return tf.where(x >= 0, x, alpha * x)


def prelu(x, name='prelu'):
    # channel-wise variant
    with tf.variable_scope(name):
        alphas = tf.get_variable(name, x.shape[-1], tf.float32, zeros)
        return tf.where(x >= 0, x, alphas * x)


def prelu2(x, name='prelu2'):
    # channel-shared variant
    with tf.variable_scope(name):
        alpha = tf.get_variable(name, 1, tf.float32, zeros)
        return tf.where(x >= 0, x, alpha * x)


def elu(x, name='elu'):
    with tf.variable_scope(name):
        return tf.where(x >= 0, x, tf.exp(x) - 1)


def pelu(x, name='pelu'):
    # channel-wise variant
    with tf.variable_scope(name):
        alphas = tf.get_variable(name, x.shape[-1], tf.float32, zeros)
        return tf.where(x >= 0, x, alphas * (tf.exp(x) - 1))


def pelu2(x, name='pelu2'):
    # channel-shared variant
    with tf.variable_scope(name):
        alpha = tf.get_variable(name, 1, tf.float32, zeros)
        return tf.where(x >= 0, x, alpha * (tf.exp(x) - 1))


def selu(x, name='selu'):
    # https://arxiv.org/abs/1706.02515
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0, x, alpha * tf.nn.elu(x))


def gelu(x, name='gelu'):
    # https://arxiv.org/abs/1606.08415
    with tf.variable_scope(name):
        return .5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + .044715 * tf.pow(x, 3)))) # noqa


def swish(x, name='swish'):
    # https://arxiv.org/abs/1710.05941
    with tf.variable_scope(name):
        return x * tf.nn.sigmoid(x)


alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
classes = len(alphabet)

char_to_int = {c: i for i, c in enumerate(alphabet)}
int_to_char = {i: c for c, i in char_to_int.items()}

x_train = [char_to_int[c] for c in alphabet[:25]]
x_train = np.reshape(x_train, [-1, 1, 1])
x_train = x_train / classes

y_train = [char_to_int[c] for c in alphabet[1:]]
y_train = np.identity(classes)[y_train]

hidden_layers = 3
cells_per_layer = 16
keep_cell_prob = 0.9
assert 0 < keep_cell_prob <= 1
l2_reg = 1e-4

epochs = 500
batch_size = 1
assert x_train.shape[0] % batch_size == 0
batches = x_train.shape[0] / batch_size

min_lr = 0.001  # default Adam learning rate
max_lr = 0.01   # x10 = 96%
step_lr = epochs / 10
scale_lr = True

training = tf.placeholder_with_default(False, [])
keep_prob = tf.where(training, keep_cell_prob, 1)

lr = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 1, 1])
y = tf.placeholder(tf.float32, [None, classes])

layers = [tf.nn.rnn_cell.GRUCell(cells_per_layer, activation=tf.nn.elu)
          for _ in range(hidden_layers)]
layers = [tf.nn.rnn_cell.DropoutWrapper(layer, output_keep_prob=keep_prob)
          for layer in layers]
rnn = tf.nn.rnn_cell.MultiRNNCell(layers)
initial_state = tuple([tf.truncated_normal(tf.shape(state), stddev=.01)
                       for state in rnn.zero_state(batch_size, tf.float32)])
yhat, _ = tf.nn.dynamic_rnn(rnn, inputs=x, initial_state=initial_state)
yhat = tf.layers.dense(yhat, classes)
yhat = tf.reshape(yhat, [batch_size, -1])

l2_loss = []
for v in tf.trainable_variables():
    if '/kernel:' in v.name:
        print('[92m+ {}[0m'.format(v.name))
        l2_loss.append(tf.nn.l2_loss(v))
    else:
        print('[90m  {}[0m'.format(v.name))
l2_loss = tf.add_n(l2_loss) * l2_reg

loss = l2_loss + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=yhat,
    labels=tf.stop_gradient(y),
))
train = tf.train.AdamOptimizer(lr).minimize(loss)

correct_predictions = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


summary_lr = tf.summary.scalar('LR', lr)
summary_loss = tf.summary.scalar('Loss', loss)
summary_accuracy = tf.summary.scalar('Accuracy', accuracy)

trainables = [tf.summary.histogram(v.name, v)
              for v in tf.trainable_variables()]

train_summaries = [summary_lr, summary_loss, summary_accuracy] + trainables
test_summaries = [summary_loss, summary_accuracy] + trainables

merged_train = tf.summary.merge(train_summaries)
merged_test = tf.summary.merge(test_summaries)

train_ops = [train, loss, accuracy, merged_train]
test_ops = [merged_test]


saver = tf.train.Saver()
for v in [loss, accuracy, y]:
    tf.add_to_collection('evaluations', v)
for v in [yhat, x]:
    tf.add_to_collection('predictions', v)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()

    train_writer = tf.summary.FileWriter('{}/train'.format(tflogs), sess.graph)
    test_writer = tf.summary.FileWriter('{}/test'.format(tflogs))

    i = 0
    j = 0
    for epoch in range(epochs):

        # https://arxiv.org/abs/1506.01186
        lr_cycle = floor(1 + epoch / (2 * step_lr))
        lr_x = abs(epoch / step_lr - 2 * lr_cycle + 1)
        lr_scale = 1 / (2 ** (lr_cycle - 1)) if scale_lr else 1
        _lr = min_lr + (max_lr - min_lr) * max(0, 1 - lr_x) * lr_scale

        # training
        avg_loss = 0
        avg_accuracy = 0
        for n in range(x_train.shape[0]):
            _, l, a, m = sess.run(train_ops, {
                x: [x_train[n]],
                y: [y_train[n]],
                training: True,
                lr: _lr,
            })
            avg_loss += l
            avg_accuracy += a
            train_writer.add_summary(m, i)
            i += 1
        avg_loss /= batches
        avg_accuracy /= batches
        print('Epoch {:5d} | Loss {:.9f} | Accuracy {:.9f}'
              .format(epoch, avg_loss, avg_accuracy))

        # testing
        for n in range(x_train.shape[0]):
            m, = sess.run(test_ops, {
                x: [x_train[n]],
                y: [y_train[n]],
            })
            test_writer.add_summary(m, j)
            j += 1

    test_writer.close()
    train_writer.close()

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

    saver.save(sess, './model')

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

print('tensorboard --logdir=train:/tmp/tflogs/train,test:/tmp/tflogs/test')
