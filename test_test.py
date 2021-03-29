#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:12:50 2019

@author: shengbang
"""
import os
from os import path
#import tensorlayer as tl
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib
from tensorflow.python.framework import ops
import pickle
from mislnet_mri import *
from tensorflow.python.tools import inspect_checkpoint as chkp
import matplotlib.pyplot as plt

tf.reset_default_graph()

TFRECORD_PATH = '/data/shengbang/MRI_128_4_ALL_MIXED/'                         

tfrecords_test = TFRECORD_PATH + 'MRI_4_Class_TEST_MIXED_20k.tfrecords'
SAMPLE_NUMBER = 20000
 
 
#test_data_iter = test_set_0.make_initializable_iterator()
#el_val = test_data_iter.get_next()

model_NUM = '253750'
acc_tot = 0
chkpt = '/data/shengbang/tensorflow_MRI/mislnet_MRI_128_4_MIXED_80k_standard/model/model-' + model_NUM
sess = tf.Session()
# Restore the model
saver = tf.train.import_meta_graph(chkpt +'.meta')
saver.restore(sess, chkpt)

# make a function to restore the method to check variables values.
def check_net_variables():
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    weights = sess.run(variables[0])
    for i in range(3):
        filters = weights[:,:,:,i] * 128 + 128
        filters = np.array(filters)
        # show the constraint layers' values
        plt.figure()
        plt.imshow(np.array(filters)[:,:,0].astype(int), cmap='gray', vmin=0, vmax=255);
        plt.show()

#sess.run(test_data_iter.initializer)
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x_placeholder:0")
y = graph.get_tensor_by_name("y_placeholder:0")
bn_phase = graph.get_tensor_by_name("bn_phase_placeholder:0")

# see mid result
mid_img = tf.get_collection("bn3")[0]
model_output = tf.get_collection("model_output")[0]
prediction = tf.cast(tf.argmax(model_output, 1), tf.int32)
result = tf.get_collection("accuracy")[0]


TEST_CASE_LIST = [0]

test_data = tf.data.TFRecordDataset(tfrecords_test).map(lambda x:read_decode(x, 1)).batch(batch_size=1)
test_data_iter = test_data.make_initializable_iterator()
el_val = test_data_iter.get_next()
sess.run(test_data_iter.initializer)
count_list = np.zeros((4, 4))
for ii, case in enumerate(TEST_CASE_LIST):
    print(ii, case)
    sys.stdout.flush()
    count_list = np.zeros((4, 4))
    
    logits = []
    labels = []
    for iter_ in range(SAMPLE_NUMBER):
        sys.stdout.flush()
        test_ims, test_lbs = sess.run(el_val)
       
        predictions = sess.run(prediction, feed_dict={x:test_ims,bn_phase:0})
        logits.append(predictions[0])
        labels.append(test_lbs[0])

    print('length of logits:', len(labels), labels[0:2])
    print('length of labels:', len(logits))
    for i in range(SAMPLE_NUMBER):
        count_list[labels[i]][logits[i]] += 1
    #print(count_list)
    sys.stdout.flush()
    np.set_printoptions(precision=4, suppress=True)
    print('Fusion Matrix: \n', count_list)
    sys.stdout.flush()    
    del logits
    del labels
#check_net_variables()
#print(labels, logits[0])

sess.close()
