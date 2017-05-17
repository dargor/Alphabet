#! /usr/bin/env python3

import numpy as np
import tensorflow as tf

# for reproducibility
np.random.seed(42) # noqa
tf.set_random_seed(42) # noqa

from functools import reduce
from random import shuffle
from shutil import rmtree
from math import floor


zeros = tf.zeros_initializer()


def model_parameters():
    def variable_parameters(v):
        return reduce(lambda x, y: x * y, v.get_shape().as_list(), 1)
    return sum(variable_parameters(v) for v in tf.trainable_variables())


def relu(x, name='relu'):
    with tf.variable_scope(name):
        zeros = tf.constant(0, tf.float32, x.shape)
        return tf.where(x < 0, zeros, x)


def lrelu(x, alpha=0.01, name='lrelu'):
    with tf.variable_scope(name):
        return tf.where(x < 0, alpha * x, x)


def prelu(x, name='prelu'):
    # channel-wise variant
    with tf.variable_scope(name):
        alphas = tf.get_variable(name, x.shape[-1], tf.float32, zeros)
        return tf.where(x < 0, alphas * x, x)


def prelu2(x, name='prelu2'):
    # channel-shared variant
    with tf.variable_scope(name):
        alpha = tf.get_variable(name, 1, tf.float32, zeros)
        return tf.where(x < 0, alpha * x, x)


def elu(x, name='elu'):
    with tf.variable_scope(name):
        return tf.where(x < 0, tf.exp(x) - 1, x)


def pelu(x, name='pelu'):
    # channel-wise variant
    with tf.variable_scope(name):
        alphas = tf.get_variable(name, x.shape[-1], tf.float32, zeros)
        return tf.where(x < 0, alphas * (tf.exp(x) - 1), x)


def pelu2(x, name='pelu2'):
    # channel-shared variant
    with tf.variable_scope(name):
        alpha = tf.get_variable(name, 1, tf.float32, zeros)
        return tf.where(x < 0, alpha * (tf.exp(x) - 1), x)


alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
classes = len(alphabet)

char_to_int = {c: i for i, c in enumerate(alphabet)}
int_to_char = {i: c for c, i in char_to_int.items()}

x_train = [char_to_int[c] for c in alphabet[:25]]
x_train = np.reshape(x_train, [len(x_train), 1, 1])
x_train = x_train / classes

y_train = [char_to_int[c] for c in alphabet[1:]]
y_train = np.identity(classes)[y_train]

hidden_layers = 3
cells_per_layer = 16
keep_cell_prob = 0.9
assert 0 < keep_cell_prob <= 1

epochs = 500
batch_size = 1
assert x_train.shape[0] % batch_size == 0
batches = x_train.shape[0] / batch_size

min_lr = 0.001  # default Adam learning rate
max_lr = 0.01   # x10 = 96%
step_lr = epochs / 10
scale_lr = True

lr = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 1, 1])
y = tf.placeholder(tf.float32, [None, classes])


class CustomLSTMCell(tf.contrib.rnn.BasicLSTMCell):

    def __call__(self, inputs, state, scope=None):
        new_h, new_state = super().__call__(inputs, state, scope)
        #                            # Prms | Loss        | Accuracy
        # None                       # 5818 | 1.258877704 | 0.640000000
        # new_h = tf.tanh(new_h)     # 5818 | 1.436888907 | 0.320000000
        # new_h = tf.nn.relu(new_h)  # 5818 | 1.953746443 | 0.240000000
        # new_h = tf.nn.elu(new_h)   # 5818 | 1.412654004 | 0.440000000
        # new_h = relu(new_h)        # 5818 | 1.724156370 | 0.360000000
        # new_h = lrelu(new_h)       # 5818 | 1.711549091 | 0.320000000
        new_h = prelu(new_h)         # 5866 | 1.243180941 | 0.560000000
        # new_h = prelu2(new_h)      # 5821 | 1.435066273 | 0.480000000
        # new_h = elu(new_h)         # 5818 | 1.500685329 | 0.400000000
        # new_h = pelu(new_h)        # 5866 | 1.354774411 | 0.640000000
        # new_h = pelu2(new_h)       # 5821 | 1.819680073 | 0.400000000
        new_h = tf.nn.dropout(new_h, keep_prob)
        return new_h, new_state


# XXX version A : vanilla LSTM cell + dropout wrapper
#     available cells    : BasicRNNCell BasicLSTMCell GRUCell LSTMCell
#                          CoupledInputForgetGateLSTMCell TimeFreqLSTMCell
# layer = tf.contrib.rnn.BasicLSTMCell(cells_per_layer)
#     available wrappers : AttentionCellWrapper DropoutWrapper EmbeddingWrapper
# layer = tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=keep_prob)

# XXX version B : custom LSTM cell, integrating prelu & dropout
layers = [CustomLSTMCell(cells_per_layer) for _ in range(hidden_layers)]

rnn = tf.contrib.rnn.MultiRNNCell(layers)
initial_state = rnn.zero_state(batch_size, tf.float32)
yhat, state = tf.nn.dynamic_rnn(rnn, inputs=x, initial_state=initial_state)
#                          # Prms | Loss        | Accuracy
# None                     # 5818 | 1.258877704 | 0.640000000
# yhat = tf.tanh(yhat)     # 5818 | 1.456664188 | 0.480000000
# yhat = tf.nn.relu(yhat)  # 5818 | 1.677613134 | 0.320000000
# yhat = tf.nn.elu(yhat)   # 5818 | 1.422254307 | 0.480000000
# yhat = relu(yhat)        # 5818 | 1.660809402 | 0.360000000
# yhat = lrelu(yhat)       # 5818 | 1.527092683 | 0.400000000
# yhat = prelu(yhat)       # 5834 | 1.387660303 | 0.400000000
# yhat = prelu2(yhat)      # 5819 | 1.383134220 | 0.440000000
# yhat = elu(yhat)         # 5818 | 1.402197974 | 0.320000000
# yhat = pelu(yhat)        # 5834 | 1.383843751 | 0.440000000
# yhat = pelu2(yhat)       # 5819 | 1.133378794 | 0.560000000
yhat = tf.layers.dense(yhat, classes)
yhat = tf.reshape(yhat, [batch_size, -1])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=yhat,
    labels=y,
))
train = tf.train.AdamOptimizer(lr).minimize(loss)

correct_predictions = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


summary_lr = tf.summary.scalar('LR', lr)
summary_loss = tf.summary.scalar('Loss', loss)
summary_accuracy = tf.summary.scalar('Accuracy', accuracy)

merged_train = tf.summary.merge([summary_lr, summary_loss, summary_accuracy])
merged_test = tf.summary.merge([summary_loss, summary_accuracy])

train_ops = [train, state, loss, accuracy, merged_train]
test_ops = [state, merged_test]


saver = tf.train.Saver()
tflogs = '/tmp/tflogs/'
rmtree(tflogs, ignore_errors=True)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    train_writer = tf.summary.FileWriter('{}/train'.format(tflogs), sess.graph)
    test_writer = tf.summary.FileWriter('{}/test'.format(tflogs))
    # to restore a previously saved model, run this instead :
    # saver.restore(sess, tf.train.latest_checkpoint('./'))

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
        s = sess.run(initial_state)
        for n in range(x_train.shape[0]):
            _, s, l, a, m = sess.run(train_ops, {
                x: [x_train[n]],
                y: [y_train[n]],
                # initial_state: s,  # keep states between batches (stateful)
                keep_prob: keep_cell_prob,
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
        s = sess.run(initial_state)
        for n in range(x_train.shape[0]):
            s, m = sess.run(test_ops, {
                x: [x_train[n]],
                y: [y_train[n]],
                # initial_state: s,  # keep states between batches (stateful)
                keep_prob: 1,
            })
            test_writer.add_summary(m, j)
            j += 1

    test_writer.close()
    train_writer.close()

    # evaluation
    avg_loss = 0
    avg_accuracy = 0
    s = sess.run(initial_state)
    for n in range(x_train.shape[0]):
        s, l, a = sess.run([state, loss, accuracy], {
            x: [x_train[n]],
            y: [y_train[n]],
            # initial_state: s,  # keep states between batches (stateful)
            keep_prob: 1,
        })
        avg_loss += l
        avg_accuracy += a
    avg_loss /= batches
    avg_accuracy /= batches
    print('Model loss: {:.9f}'.format(avg_loss))
    print('Model accuracy: {:.9f}'.format(avg_accuracy))
    print('Model parameters: {}'.format(model_parameters()))

    saver.save(sess, 'model')

    # predictions
    l = list(alphabet[:25])
    shuffle(l)
    s = sess.run(initial_state)
    for a in l:
        # prepare input data
        _x = np.reshape([char_to_int[a]], [1, 1]) / classes
        # feed prepared input data to the model
        pred_y = yhat.eval({
            x: [_x],
            # initial_state: s,  # keep states between batches (stateful)
            keep_prob: 1,
        })
        # convert output back to something lisible
        real_y = int_to_char[np.argmax(pred_y)]
        # check if output match our expectations
        expected = chr(ord(a) + 1)
        print('{} -> {} {}'
              .format(a, real_y, '' if real_y == expected else '*'))

print('tensorboard --logdir=train:/tmp/tflogs/train,test:/tmp/tflogs/test')
