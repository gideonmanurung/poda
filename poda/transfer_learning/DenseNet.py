# Copyright 2016 pudae. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition of the DenseNet architecture.
As described in https://arxiv.org/abs/1608.06993.
  Densely Connected Convolutional Networks
  Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

from poda.layers.activation import *
from poda.layers.dense import *

@slim.add_arg_scope
def _global_avg_pool2d(inputs, data_format='NHWC', scope=None, outputs_collections=None):
  with tf.variable_scope(scope, 'xx', [inputs]) as sc:
    axis = [1, 2] if data_format == 'NHWC' else [2, 3]
    net = tf.reduce_mean(inputs, axis=axis, keepdims=True)
    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net


@slim.add_arg_scope
def _conv(inputs, num_filters, kernel_size, stride=1, dropout_rate=None,
          scope=None, outputs_collections=None):
  with tf.variable_scope(scope, 'xx', [inputs]) as sc:
    net = slim.batch_norm(inputs)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, num_filters, kernel_size)

    if dropout_rate:
      net = tf.nn.dropout(net)

    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  return net


@slim.add_arg_scope
def _conv_block(inputs, num_filters, data_format='NHWC', scope=None, outputs_collections=None):
  with tf.variable_scope(scope, 'conv_blockx', [inputs]) as sc:
    net = inputs
    net = _conv(net, num_filters*4, 1, scope='x1')
    net = _conv(net, num_filters, 3, scope='x2')
    if data_format == 'NHWC':
      net = tf.concat([inputs, net], axis=3)
    else: # "NCHW"
      net = tf.concat([inputs, net], axis=1)

    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  return net


@slim.add_arg_scope
def _dense_block(inputs, num_layers, num_filters, growth_rate,
                 grow_num_filters=True, scope=None, outputs_collections=None):

  with tf.variable_scope(scope, 'dense_blockx', [inputs]) as sc:
    net = inputs
    for i in range(num_layers):
      branch = i + 1
      net = _conv_block(net, growth_rate, scope='conv_block'+str(branch))

      if grow_num_filters:
        num_filters += growth_rate

    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  return net, num_filters


@slim.add_arg_scope
def _transition_block(inputs, num_filters, compression=1.0,
                      scope=None, outputs_collections=None):

  num_filters = int(num_filters * compression)
  with tf.variable_scope(scope, 'transition_blockx', [inputs]) as sc:
    net = inputs
    net = _conv(net, num_filters, 1, scope='blk')

    net = slim.avg_pool2d(net, 2)

    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  return net, num_filters


def build_top_layer_model(base_layer,num_classes,num_depthwise_layer=None,
                          num_fully_connected_layer=1,num_hidden_unit=512,
                          activation_fully_connected='relu',dropout_keep_prob=None,regularizers=None,num_classes=1000):
    previous_layer = base_layer
    if num_depthwise_layer != None:
        num_depthwise_layer = num_depthwise_layer * 3
        for i in range(num_depthwise_layer):
            depth_wise_net = depthwise_convolution_2d(input_tensor=previous_layer,number_filters=base_layer.shape[3], 
                                                      kernel_size=(3,3), stride_size=(2,2), padding='same',
                                                      activation_function='relu',name='depthwise_conv2d_'+str(i))
            previous_layer = depth_wise_net
    else:
        depth_wise_net = previous_layer

    flatten_layer = flatten(input_tensor=depth_wise_net)

    if num_fully_connected_layer != None:
        for j in range(num_fully_connected_layer):
            fully_connected_net = fully_connected(input_tensor=flatten_layer,hidden_unit=num_hidden_unit,
                                                  activation_function=activation_fully_connected,
                                                  dropout_layer=dropout_keep_prob,regularizers=regularizers,
                                                  scale=dropout_keep_prob,name='fully_connected_'+str(j))
            flatten_layer = fully_connected_net
    else:
        flatten_layer = flatten_layer

    non_logit = fully_connected(input_tensor=flatten_layer,hidden_unit=num_classes,activation_function=None)
    if num_classes > 2:
      output = softmax(input_tensor=non_logit)
    else:
      output = sigmoid(input_tensor=non_logit)
    return non_logit, output

def densenet(inputs,
             num_classes=1000,
             reduction=None,
             growth_rate=None,
             num_filters=None,
             num_layers=None,
             dropout_rate=None,
             data_format='NHWC',
             is_training=True,
             reuse=None,
             scope=None):
  assert reduction is not None
  assert growth_rate is not None
  assert num_filters is not None
  assert num_layers is not None

  compression = 1.0 - reduction
  num_dense_blocks = len(num_layers)

  if data_format == 'NCHW':
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

  with tf.variable_scope(scope, 'densenetxxx', [inputs, num_classes],
                         reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                         is_training=is_training), \
         slim.arg_scope([slim.conv2d, _conv, _conv_block,
                         _dense_block, _transition_block], 
                         outputs_collections=end_points_collection), \
         slim.arg_scope([_conv], dropout_rate=dropout_rate):
      net = inputs

      # initial convolution
      net = slim.conv2d(net, num_filters, 7, stride=2, scope='conv1')
      net = slim.batch_norm(net)
      net = tf.nn.relu(net)
      net = slim.max_pool2d(net, 3, stride=2, padding='SAME')

      # blocks
      for i in range(num_dense_blocks - 1):
        # dense blocks
        net, num_filters = _dense_block(net, num_layers[i], num_filters,
                                        growth_rate,
                                        scope='dense_block' + str(i+1))

        # Add transition_block
        net, num_filters = _transition_block(net, num_filters,
                                             compression=compression,
                                             scope='transition_block' + str(i+1))

      net, num_filters = _dense_block(
              net, num_layers[-1], num_filters,
              growth_rate,
              scope='dense_block' + str(num_dense_blocks))

  var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

  non_logit , top_layer = build_top_layer_model(net,num_classes=num_classes,num_depthwise_layer=num_depthwise_layer,
                                                    num_fully_connected_layer=num_fully_connected_layer,
                                                    num_hidden_unit=num_hidden_unit,
                                                    activation_fully_connected=activation_fully_connected,regularizers=regularizers)

  return var_list , non_logit , top_layer


def densenet121(inputs, num_classes=1000, data_format='NHWC', is_training=True, reuse=None,
                num_depthwise_layer=None,num_fully_connected_layer=None,
                num_hidden_unit=None,activation_fully_connected=None,
                regularizers=None):
  return densenet(inputs,
                  num_classes=num_classes, 
                  reduction=0.5,
                  growth_rate=32,
                  num_filters=64,
                  num_layers=[6,12,24,16],
                  data_format=data_format,
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet121')
densenet121.default_image_size = 224


def densenet161(inputs, num_classes=1000, data_format='NHWC', is_training=True, reuse=None,
                num_depthwise_layer=None,num_fully_connected_layer=None,
                num_hidden_unit=None,activation_fully_connected=None,
                regularizers=None):
  return densenet(inputs,
                  num_classes=num_classes, 
                  reduction=0.5,
                  growth_rate=48,
                  num_filters=96,
                  num_layers=[6,12,36,24],
                  data_format=data_format,
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet161')
densenet161.default_image_size = 224


def densenet169(inputs, num_classes=1000, data_format='NHWC', is_training=True, reuse=None,
                num_depthwise_layer=None,num_fully_connected_layer=None,
                num_hidden_unit=None,activation_fully_connected=None,
                regularizers=None):
  return densenet(inputs,
                  num_classes=num_classes, 
                  reduction=0.5,
                  growth_rate=32,
                  num_filters=64,
                  num_layers=[6,12,32,32],
                  data_format=data_format,
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet169')
densenet169.default_image_size = 224


def densenet_arg_scope(weight_decay=1e-4,
                       batch_norm_decay=0.99,
                       batch_norm_epsilon=1.1e-5,
                       data_format='NHWC'):
  with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.avg_pool2d, slim.max_pool2d,
                       _conv_block, _global_avg_pool2d],
                      data_format=data_format):
    with slim.arg_scope([slim.conv2d],
                         weights_regularizer=slim.l2_regularizer(weight_decay),
                         activation_fn=None,
                         biases_initializer=None):
      with slim.arg_scope([slim.batch_norm],
                          scale=True,
                          decay=batch_norm_decay,
                          epsilon=batch_norm_epsilon) as scope:
        return scope