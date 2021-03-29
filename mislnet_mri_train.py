#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:54:24 2019

@author: shengbang
"""

import os
from os import path
#import tensorlayer as tl
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib
from tensorflow.python.framework import ops
from mislnet_mri import *
import math

# This script is in lab24

print(tf.VERSION)
tf.reset_default_graph()

# Project naming and parameters setting
exp_name = 'mislnet_MRI_128_2_MIXED_LESS_80k'
model_scope = 'mislnet_MRI_2_Class'
ep = 200
ep_decay = 5

weight_decay = 0.001
learning_rate = 0.00001
lr_decay = 0.8

cls_num = 2
tot_patch = int(40000 * cls_num)
tot_patch_test = int(2000 * cls_num)
channel = 1
im_size = 128
batch_size = 64
test_size = 50
test_iter = int(tot_patch_test/test_size)
generations = int(ep * tot_patch/batch_size)
stepsize = int(ep_decay * tot_patch/batch_size)

eval_every = int(tot_patch/batch_size)
GLOBAL_STEP = 0

save_dir = '/data/shengbang/tensorflow_MRI/'
save_path = path.join(save_dir, exp_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)

log_path = path.join(save_path, 'tflog')
if not os.path.exists(log_path):
    os.mkdir(log_path)
tfboard = path.join(save_path, 'tfboard')
if not os.path.exists(tfboard):
    os.mkdir(tfboard)

trained_model_path = path.join(save_path, 'model')
if not os.path.exists(trained_model_path):
    os.mkdir(trained_model_path)


def bar_f(x):
    if(abs(x)>1):
        return np.sign(x)
    else:
        return x
vfunc = np.vectorize(bar_f)

def sigmoid(x):
    return 1/(1+(math.e**-x))


def constrain(w):
    w = w * 10000
    w[2, 2, :, :] = 0
    w = w.reshape([1, 25, channel, 3])
    w = w / w.sum(1)
    w = w.reshape([5, 5, channel, 3])

    # prevent w from explosion
    #w = sigmoid(w) * 2 - 1
    w = vfunc(w)

    w[2, 2, :, :] = -1
    return w




def get_loss(logits, targets):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#    tot_loss = cross_entropy_mean + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    #weights_norm = weight_decay * tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection('weights')])
    #weights_norm_mean = tf.reduce_sum(weights_norm)
    #total_loss = cross_entropy_mean + weights_norm_mean
    #return (cross_entropy_mean)
    return cross_entropy_mean

def train_step(loss_value):
    #model_learning_rate = tf.train.exponential_decay(learning_rate, global_step, stepsize, lr_decay, staircase=True)
    #my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
    my_optimizer = tf.train.AdamOptimizer(learning_rate)
    #my_optimizer = tf.train.MomentumOptimizer(model_learning_rate,0.9)
    train_step = my_optimizer.minimize(loss_value, global_step=global_step)
    return (train_step)


def accuracy_of_batch(logits, targets):
    labels = tf.cast(targets, tf.int32)
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    predicted_correctly = tf.equal(batch_predictions, labels)
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return (accuracy)


#initialize dataset and iterator, return image and labels
    
TFRECORD_PATH = '/data/shengbang/MRI_128_2_ALL_MIXED/'

tfrecords_train = TFRECORD_PATH + 'MRI_2_Class_TRAIN_MIXED_80k.tfrecords'

tfrecords_test = TFRECORD_PATH + 'MRI_2_Class_VALID_MIXED_4k.tfrecords'

dataset = tf.data.TFRecordDataset([tfrecords_train]).map(lambda x: read_decode(x, 1)).repeat().batch(batch_size=batch_size)
dataset_test = tf.data.TFRecordDataset([tfrecords_test]).map(lambda x: read_decode(x, 1)).repeat().batch(batch_size=test_size)

data_iter = dataset.make_initializable_iterator()
test_data_iter = dataset_test.make_initializable_iterator()
el = data_iter.get_next()
el_val = test_data_iter.get_next()

# input vector for images and labels
x = tf.placeholder(dtype=tf.float32,shape=[None,im_size,im_size,channel],name='x_placeholder')
y = tf.placeholder(dtype=tf.int64,shape=[None,],name= 'y_placeholder')
bn_phase = tf.placeholder(tf.bool,name='bn_phase_placeholder')

model_output = mislnet(x, False, bn_phase, cls_num, model_scope)
tf.add_to_collection('model_output',model_output)

loss = get_loss(model_output, y)
tf.add_to_collection('loss',loss)
train_summ = tf.summary.scalar("loss", loss)

accuracy = accuracy_of_batch(model_output, y)
tf.add_to_collection('accuracy',accuracy)

test_acc = tf.placeholder(tf.float32, shape=(),name='test_acc_placeholder')
test_summ = tf.summary.scalar('test_acc', test_acc)

global_step = tf.Variable(GLOBAL_STEP, trainable=False)
tf.add_to_collection('global_step',global_step)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = train_step(loss)
tf.add_to_collection('train_op',train_op)

convF_placeholder = tf.placeholder(tf.float32, shape=[5,5,channel,3],name='convF_placeholder')
convF_w = tf.get_collection('convF_w')[0]
constrain_op = convF_w.assign(convF_placeholder)

print('Initializing the Variables.')
sys.stdout.flush()
init_op = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=1)
save_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope)

#testing_summ = []
acc_tuple = []
loss_tuple = []
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(data_iter.initializer)
    sess.run(test_data_iter.initializer)
    
    print('Starting Training')
    sys.stdout.flush()

    summary_writer = tf.summary.FileWriter(tfboard, tf.get_default_graph())
    summary_writer.reopen()
    w = sess.run(convF_w)
    w = constrain(w)
    sess.run([constrain_op], {convF_placeholder: w})

    sess.graph.finalize()
    for i in range(generations):
    #for i in range(5):
        itr = sess.run(global_step)
        #print itr

        ims, lbs = sess.run(el)
#        print(ims.shape, lbs.shape)
        sys.stdout.flush()
        _, loss_value, training_summ = sess.run([train_op, loss,train_summ],feed_dict={x:ims,y:lbs,bn_phase:1})
        summary_writer.add_summary(training_summ, global_step=itr)
        output = 'Iter {}/{}: Loss = {:.5f}'.format(itr,generations, loss_value)
        print(output)

        w = sess.run(convF_w)
        w = constrain(w)
        sess.run([constrain_op], {convF_placeholder: w})

        if (i + 1) % eval_every == 0:
            saver.save(sess, trained_model_path + '/model', global_step=global_step)

            #tl.files.save_npz_dict(save_list=save_list, name=trained_model_path + '/model-'+str(i+1)+'.npz', sess=sess)

            acc_tot = 0

            for ii in range(test_iter):
                #test_ims, test_lbs = sess.run([test_images, test_targets])
                test_ims, test_lbs = sess.run(el_val)
                
                temp_accuracy = sess.run(accuracy,feed_dict={x:test_ims,y:test_lbs,bn_phase:0})
                acc_tot = acc_tot + temp_accuracy
            acc_tot = acc_tot / test_iter
        #    testing_summ.append(acc_tot)
            acc_tuple.append(acc_tot)
            loss_tuple.append(loss_value)
            testing_summ = sess.run(test_summ, feed_dict={test_acc: acc_tot})
            summary_writer.add_summary(testing_summ, global_step=itr)
            acc_output = ' --- Test Accuracy = {:.2f}%.'.format(100. * acc_tot)
            print(acc_output)
        del ims,lbs,output,w,itr
