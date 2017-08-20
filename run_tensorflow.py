#! /usr/bin/env python3

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


#                                     # Prms | Loss        | Accuracy
# tf.nn.rnn_cell.BasicRNNCell         # 1834 | 0.840756910 | 0.920000000
# tf.nn.rnn_cell.GRUCell              # 4522 | 0.645110278 | 1.000000000
# tf.nn.rnn_cell.BasicLSTMCell        # 5866 | 0.737078576 | 1.000000000
# tf.nn.rnn_cell.LSTMCell             # 5866 | 0.737078576 | 1.000000000
# tf.nn.rnn_cell.LSTMCell + peepholes # 6010 | 0.745531615 | 0.920000000
class CustomRNNCell(tf.nn.rnn_cell.GRUCell):

    def __call__(self, inputs, state, scope=None):
        new_h, new_state = super().__call__(inputs, state, scope)
        #                            # Prms | Loss        | Accuracy
        # None                       # 5818 | 0.765377985 | 0.920000000
        # new_h = tf.nn.relu(new_h)  # 5818 | 1.631359415 | 0.560000000
        # new_h = tf.nn.elu(new_h)   # 5818 | 0.810860920 | 0.880000000
        # new_h = relu(new_h)        # 5818 | 1.805080099 | 0.520000000
        # new_h = lrelu(new_h, 0.01) # 5818 | 1.375006690 | 0.680000000
        # new_h = lrelu(new_h, 0.2)  # 5818 | 0.968093691 | 0.880000000
        new_h = prelu(new_h)         # 5866 | 0.737078576 | 1.000000000
        # new_h = prelu2(new_h)      # 5821 | 0.752888067 | 0.960000000
        # new_h = elu(new_h)         # 5818 | 0.821999593 | 0.800000000
        # new_h = pelu(new_h)        # 5866 | 0.710354111 | 0.960000000
        # new_h = pelu2(new_h)       # 5821 | 0.601579508 | 0.920000000
        # new_h = selu(new_h)        # 5818 | 0.780245475 | 0.880000000
        new_h = tf.nn.dropout(new_h, keep_prob)
        return new_h, new_state


layers = [CustomRNNCell(cells_per_layer) for _ in range(hidden_layers)]
rnn = tf.nn.rnn_cell.MultiRNNCell(layers)
initial_state = rnn.zero_state(batch_size, tf.float32)
yhat, _ = tf.nn.dynamic_rnn(rnn, inputs=x, initial_state=initial_state)
yhat = tf.layers.dense(yhat, classes)
yhat = tf.reshape(yhat, [batch_size, -1])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=yhat,
    labels=y,
))
# XXX non adaptive optimizers
# optimizer = tf.train.GradientDescentOptimizer(lr)
# optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=False)
# optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=True)
# XXX adaptive optimizers
# optimizer = tf.train.AdadeltaOptimizer(lr)
# optimizer = tf.train.AdagradOptimizer(lr)
# optimizer = tf.train.RMSPropOptimizer(lr)
optimizer = tf.train.AdamOptimizer(lr)
# Optimizer | Loss        | Accuracy
# SGD       | 3.235350876 | 0.040000000
# Momentum  | 3.221479120 | 0.040000000
# NAG       | 3.221476517 | 0.040000000
# Adadelta  | 3.256221209 | 0.080000000
# Adagrad   | 3.240393400 | 0.040000000
# RMSProp   | 1.111033341 | 0.680000000
# Adam      | 0.737078576 | 1.000000000
train = optimizer.minimize(loss)

correct_predictions = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


summary_lr = tf.summary.scalar('LR', lr)
summary_loss = tf.summary.scalar('Loss', loss)
summary_accuracy = tf.summary.scalar('Accuracy', accuracy)

merged_train = tf.summary.merge([summary_lr, summary_loss, summary_accuracy])
merged_test = tf.summary.merge([summary_loss, summary_accuracy])

train_ops = [train, loss, accuracy, merged_train]
test_ops = [merged_test]


saver = tf.train.Saver()
for v in [loss, accuracy, y]:
    tf.add_to_collection('evaluations', v)
for v in [yhat, x, keep_prob]:
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
        for n in range(x_train.shape[0]):
            m, = sess.run(test_ops, {
                x: [x_train[n]],
                y: [y_train[n]],
                keep_prob: 1,
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
    for a in l:
        # prepare input data
        _x = np.reshape([char_to_int[a]], [1, 1]) / classes
        # feed prepared input data to the model
        pred_y = yhat.eval({
            x: [_x],
            keep_prob: 1,
        })
        # convert output back to something lisible
        real_y = int_to_char[np.argmax(pred_y)]
        # check if output match our expectations
        expected = chr(ord(a) + 1)
        print('{} -> {} {}'
              .format(a, real_y, '' if real_y == expected else '*'))

print('tensorboard --logdir=train:/tmp/tflogs/train,test:/tmp/tflogs/test')
