#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:53:31 2019

@author: shengbang
"""

import tensorflow as tf

def mislnet(input_images,reuse,bn_phase,cls_num,model_scope):
    #tf.contrib.layers.xavier_initializer()
    #tf.truncated_normal_initializer(stddev=0.02)
    #regularizer = tf.contrib.layers.l2_regularizer(scale=weigth_decay)
    def weight_init(name, shape,init = tf.contrib.layers.xavier_initializer()):
        var = tf.get_variable(name = name, shape = shape, dtype=tf.float32,initializer=init)#,regularizer=regularizer)
        return var

    def bias_init(name, shape, val):
        var = tf.get_variable(name=name, shape=shape, dtype=tf.float32,initializer=tf.constant_initializer(val))
        return var

    def bn_tanh(inputs,name,bn_phase):
        with tf.variable_scope(name):
            inputs = tf.contrib.layers.batch_norm(inputs,center=True,scale=True,is_training=bn_phase)
            inputs = tf.nn.tanh(inputs)
        return inputs

    with tf.variable_scope(model_scope,reuse=reuse) as scope:

        # 128*128*6
        convF_w = weight_init(name='convF_w', shape=[5, 5, 1, 3])
        tf.add_to_collection('convF_w',convF_w)
        convF = tf.nn.conv2d(input_images, convF_w, [1, 1, 1, 1], padding='VALID',name='convF')
        tf.add_to_collection('convF', convF)
        # 124*124*6

        conv2_kernel = weight_init(name='conv2_kernel', shape=[7, 7, 3, 96])
        tf.add_to_collection('conv2_kernel', conv2_kernel)
        conv2_bias = bias_init(name='conv2_bias', shape=[96], val=0.0)
        tf.add_to_collection('conv2_bias', conv2_bias)
        conv2 = tf.nn.conv2d((convF), conv2_kernel, [1, 2, 2, 1], padding='VALID',name = 'conv2') + conv2_bias
        tf.add_to_collection('conv2', conv2)
        
        bn1 = bn_tanh(conv2,'bn1',bn_phase)
        tf.add_to_collection('bn1', bn1)
        # 59*59*96
        
        
        pool1 = tf.nn.max_pool(bn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name='pool1')
        tf.add_to_collection('pool1', pool1)
        
        
        # 29*29*96
        conv3_kernel = weight_init(name='conv3_kernel', shape=[5, 5, 96, 64])
        tf.add_to_collection('conv3_kernel', conv3_kernel)
        conv3_bias = bias_init(name='conv3_bias', shape=[64], val=0.0)
        tf.add_to_collection('conv3_bias', conv3_bias)
        conv3 = tf.nn.conv2d(pool1, conv3_kernel, [1, 1, 1, 1], padding='SAME', name='conv3') + conv3_bias
        tf.add_to_collection('conv3', conv3)
        bn2 = bn_tanh(conv3,'bn2',bn_phase)
        tf.add_to_collection('bn2', bn2)         
        
        
        pool2 = tf.nn.max_pool(bn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name='pool2')
        tf.add_to_collection('pool2', pool2)
           
        # 14*14*64
        conv4_kernel = weight_init(name='conv4_kernel', shape=[5, 5, 64, 64])
        tf.add_to_collection('conv4_kernel', conv4_kernel)
        conv4_bias = bias_init(name='conv4_bias', shape=[64], val=0.0)
        tf.add_to_collection('conv4_bias', conv4_bias)
        conv4 = tf.nn.conv2d(pool2, conv4_kernel, [1, 1, 1, 1], padding='SAME', name='conv4') + conv4_bias
        tf.add_to_collection('conv4', conv4)

        bn3 = bn_tanh(conv4,'bn3',bn_phase)
        tf.add_to_collection('bn3', bn3)
              
        pool3 = tf.nn.max_pool(bn3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name='pool3')
        tf.add_to_collection('pool3', pool3)


        # 6*6*128
        conv6_kernel = weight_init(name='conv6_kernel', shape=[1, 1, 64, 128])
        tf.add_to_collection('conv6_kernel', conv6_kernel)
        conv6_bias = bias_init(name='conv6_bias', shape=[128], val=0.0)
        tf.add_to_collection('conv6_bias', conv6_bias)
        conv6 = tf.nn.conv2d(pool3, conv6_kernel, [1, 1, 1, 1], padding='SAME', name='conv6') + conv6_bias
        tf.add_to_collection('conv6', conv6)

        bn5 = bn_tanh(conv6,'bn5',bn_phase)
        tf.add_to_collection('bn5', bn5)
        pool5 = tf.nn.avg_pool(bn5, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID',name='pool5')

        # 2 * 2 * 128
        reshaped_output = tf.reshape(pool5, [-1, 2 * 2 * 128])
        #reshaped_output = tf.reshape(bn4, [-1, 12*12*128])

        tf.add_to_collection('fc0', reshaped_output)

        fc1_weight = weight_init(name='fc1_weight', shape=[2 * 2 * 128, 200])
        tf.add_to_collection('fc1_weight', fc1_weight)
        fc1_bias = bias_init(name='fc1_bias', shape=[200], val=0.0)
        tf.add_to_collection('fc1_bias', fc1_bias)

        fc1 = tf.nn.tanh(tf.add(tf.matmul(reshaped_output, fc1_weight), fc1_bias))
        tf.add_to_collection('fc1', fc1)

        fc2_weight = weight_init(name='fc2_weight', shape=[200, 200])
        tf.add_to_collection('fc2_weight', fc2_weight)
        fc2_bias = bias_init(name='fc2_bias', shape=[200], val=0.0)
        tf.add_to_collection('fc2_bias', fc2_bias)

        fc2 = tf.nn.tanh(tf.add(tf.matmul(fc1, fc2_weight), fc2_bias))
        tf.add_to_collection('fc2', fc2)

        fc3_weight = weight_init(name='fc3_weight', shape=[200, cls_num])
        tf.add_to_collection('fc3_weight', fc3_weight)
        fc3_bias = bias_init(name='fc3_bias', shape=[cls_num], val=0.0)
        tf.add_to_collection('fc3_bias', fc3_bias)

        fc3 = tf.add(tf.matmul(fc2, fc3_weight), fc3_bias)

        tf.add_to_collection('fc3',fc3)

    return fc3


def read_decode(iterator, label):
    img_features = tf.parse_single_example(
            iterator,
            features={
                    'label': tf.FixedLenFeature([], tf.int64),
                    'image': tf.FixedLenFeature([], tf.string)
            })
    
    image = tf.decode_raw(img_features['image'], tf.int16)
    image = tf.reshape(image, [128, 128, 1])  
    image = tf.cast(image, tf.float32)   
    img_label = tf.cast(img_features['label'], tf.int32)
    return image, img_label


    
