import tensorflow as tf
from poda.layers.merge import *
from poda.layers.dense import *
from poda.layers.activation import *
from poda.layers.regularizer import *
from poda.layers.convolutional import *

class VGG16(object):
    def __init__(self, input_tensor, num_blocks=5, classes=1000, batch_normalizations = True):
        """[summary]
        
        Arguments:
            object {[type]} -- [description]
            input_tensor {[type]} -- [description]
        
        Keyword Arguments:
            classes {int} -- [description] (default: {1000})
            batch_normalization {bool} -- [description] (default: {True})
        """
        self.input_tensor = input_tensor
        self.classes = classes
        self.batch_normalization = batch_normalizations
        self.num_block = num_blocks

    def vgg_block(self, input_tensor, num_block, batch_normalization=True, scope=None, reuse=None):
        with tf.variable_scope(scope, 'vgg_16', [input_tensor]):
            with tf.variable_scope('Block_1'):
                conv_1 = convolution_2d(input_tensor=input_tensor, number_filters=64, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)
                conv_2 = convolution_2d(input_tensor=conv_1, number_filters=64, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)
                conv_3 = max_pool_2d(input_tensor=conv_2, pool_sizes=(2,2), stride_sizes=(2,2), paddings='valid', names=None)
            with tf.variable_scope('Block_2'):
                conv_4 = convolution_2d(input_tensor=conv_3, number_filters=128, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)
                conv_5 = convolution_2d(input_tensor=conv_4, number_filters=128, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)
                conv_6 = max_pool_2d(input_tensor=conv_5, pool_sizes=(2,2), stride_sizes=(2,2), paddings='valid', names=None)
            with tf.variable_scope('Block_3'):
                conv_7 = convolution_2d(input_tensor=conv_6, number_filters=256, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)
                conv_8 = convolution_2d(input_tensor=conv_7, number_filters=256, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)
                conv_9 = convolution_2d(input_tensor=conv_8, number_filters=256, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)
                conv_10 = max_pool_2d(input_tensor=conv_9, pool_sizes=(2,2), stride_sizes=(2,2), paddings='valid', names=None)
            with tf.variable_scope('Block_4'):
                conv_11 = convolution_2d(input_tensor=conv_10, number_filters=512, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)
                conv_12 = convolution_2d(input_tensor=conv_11, number_filters=512, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)
                conv_13 = convolution_2d(input_tensor=conv_12, number_filters=512, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)
                conv_14 = max_pool_2d(input_tensor=conv_13, pool_sizes=(2,2), stride_sizes=(2,2), paddings='valid', names=None)
            with tf.variable_scope('Block_5'):
                conv_15 = convolution_2d(input_tensor=conv_14, number_filters=512, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)
                conv_16 = convolution_2d(input_tensor=conv_15, number_filters=512, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)
                conv_17 = convolution_2d(input_tensor=conv_16, number_filters=512, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)
                conv_18 = max_pool_2d(input_tensor=conv_17, pool_sizes=(2,2), stride_sizes=(2,2), paddings='valid', names=None)

        if num_block==1:
            vgg_16 = conv_3
        elif num_block==2:
            vgg_16 = conv_6
        elif num_block==3:
            vgg_16 = conv_10
        elif num_block==4:
            vgg_16 = conv_14
        elif num_block==5:
            vgg_16 = conv_18
        else:
            vgg_16 = conv_18

        return vgg_16

    def create_model(self, input_tensor, num_depthwise_layer=None, num_dense_layer=1, num_hidden_unit=512, activation_dense='relu', dropout_rate=None, regularizer=None, classes=1000, scope=None):
        number_filter = input_tensor.get_shape().as_list()[-1]

        with tf.variable_scope(scope, 'vgg_16', [input_tensor]):
            base_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            if num_depthwise_layer==None or num_depthwise_layer<0:
                vgg_16 = dense(input_tensor=input_tensor, hidden_units=num_hidden_unit, names=None, activations=activation_dense, regularizers=regularizer)
            else:
                for i in range(0,num_depthwise_layer):
                    vgg_16 = depthwise_convolution_2d(input_tensor=input_tensor, number_filters=number_filter, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='same', activations='relu', dropout_layers=None, names=None)

            non_logit = dense(input_tensor=vgg_16, hidden_units=classes, names='output')

            if classes > 2:
                output = softmax(input_tensor=non_logit)
            else:
                output = sigmoid(input_tensor=non_logit)

            full_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return non_logit, output, base_var_list, full_var_list