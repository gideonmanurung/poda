import tensorflow as tf
from poda.layers.merge import *
from poda.layers.dense import *
from poda.layers.merge import *
from poda.layers.activation import *
from poda.layers.regularizer import *
from poda.layers.convolutional import *


class InceptionV4(object):
    def __init__(self, input_tensor, n_inception_a, n_inception_b, n_inception_c, classes=1000, batch_normalizations = True):
        """[summary]
        
        Arguments:
            num_classes {[type]} -- [description]
        
        Keyword Arguments:
            input_tensor {[type]} -- [description] (default: {None})
            input_shape {tuple} -- [description] (default: {(None, 300, 300, 3)})
            learning_rate {float} -- [description] (default: {0.0001})
            batch_normalizations {bool} -- [description] (default: {True})
        """
        self.input_tensor = input_tensor
        self.classes = classes
        self.n_inception_a = n_inception_a
        self.n_inception_b = n_inception_b
        self.n_inception_c = n_inception_c
        self.batch_normalizations = batch_normalizations

        self.create_model(input_tensor=self.input_tensor, classes=self.classes, n_inception_a=self.n_inception_a, n_inception_b=self.n_inception_b, n_inception_c=self.n_inception_c, batch_normalizations=self.batch_normalizations)


    def conv_block(self, inputs, filters, kernel_size, strides=(2,2), padding='VALID',  dropout_rate=0.2, activation=None, name=None, use_batch_norm=True):
        """[summary]
        
        Arguments:
            inputs {[type]} -- [description]
            filters {[type]} -- [description]
            kernel_size {[type]} -- [description]
        
        Keyword Arguments:
            strides {tuple} -- [description] (default: {(2,2)})
            padding {str} -- [description] (default: {'VALID'})
            dropout_rate {float} -- [description] (default: {0.2})
            activation {[type]} -- [description] (default: {None})
            name {[type]} -- [description] (default: {None})
            use_batch_norm {bool} -- [description] (default: {True})
        """
        conv = convolution_2d(input_tensor=inputs, number_filters=filters, kernel_sizes=kernel_size, stride_sizes=strides, paddings=padding, activations=activation, names=name)
        
        conv = batch_normalization(input_tensor=conv, is_trainable=use_batch_norm)
        conv = dropout(input_tensor=conv, dropout_rates=dropout_rate)
        return conv

    # Stem Block
    def stem_block(self, input_tensor, batch_normalization=True, scope=None, reuse=None):
        """[summary]
        
        Arguments:
            input_tensor {[type]} -- [description]
        
        Keyword Arguments:
            batch_normalization {bool} -- [description] (default: {True})
            scope {[type]} -- [description] (default: {None})
            reuse {[type]} -- [description] (default: {None})
        """
        with tf.variable_scope(scope, 'StemBlock', [input_tensor], reuse=reuse):
            conv_1 = self.conv_block(inputs=input_tensor, filters=32, kernel_size=(3,3), strides=(2, 2), use_batch_norm=batch_normalization, name='0_3x3')

            conv_2 = self.conv_block(inputs=conv_1, filters=32, kernel_size=(3,3), strides=(1,1), use_batch_norm=batch_normalization, name='0_3x3')
            
            conv_3 = self.conv_block(inputs=conv_2, filters=64, kernel_size=(3,3), padding='SAME', strides=(1, 1), use_batch_norm=batch_normalization, name='0_3x3')
            
            conv_4 = self.conv_block(inputs=conv_3, filters=96, kernel_size=(3,3), strides=(2,2), use_batch_norm=batch_normalization, name='0b_3x3')
            max_pool_1 = max_pool_2d(input_tensor=conv_3, pool_sizes=(3,3), stride_sizes=(2,2), paddings='VALID', names='0a_3x3')

            concat_1 = concatenate(input_tensor_1=conv_4, input_tensor_2=max_pool_1, axis=-1, names='conv_3x3_maxpool_3x3')

            conv_5 = self.conv_block(inputs=concat_1, filters=64, kernel_size=(1,1), padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='0a_1x1')
            conv_6 = self.conv_block(inputs=conv_5, filters=96, kernel_size=(3,3), strides=(1,1), use_batch_norm=batch_normalization, name='1a_3x3')

            conv_7 = self.conv_block(inputs=concat_1, filters=64, kernel_size=(3,3), padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='0b_1x1')
            conv_8 = self.conv_block(inputs=conv_7, filters=64, kernel_size=(7,1), padding='SAME', strides=(1, 1), use_batch_norm=batch_normalization, name='1b_7x1')
            conv_9 = self.conv_block(inputs=conv_8, filters=64, kernel_size=(1,7), padding='SAME', strides=(1, 1), use_batch_norm=batch_normalization, name='2b_1x7')
            conv_10 = self.conv_block(inputs=conv_9, filters=96, kernel_size=(3,3), strides=(1, 1), use_batch_norm=batch_normalization, name='3b_3x3')
            
            
            concat_2 = concatenate(input_tensor_1=conv_6, input_tensor_2=conv_10, axis=-1, names='conv_3x3_conv_3x3')

            max_pool_2 = max_pool_2d(inputs=concat_2, pool_sizes=(3,3), stride_sizes=(2,2), paddings='VALID', names='0b_3x3')
            conv_11 = self.conv_block(inputs=concat_2, filters=192, kernel_size=(3,3), strides=(2,2), use_batch_norm=batch_normalization, name='0a_3x3')
            
            concat_3 = concatenate(input_tensor_1=max_pool_2, input_tensor_2=conv_11, axis=-1, names='maxpool_3x3_conv_3x3')
        return concat_3
    
    # Inception A
    def inception_a(self, input_tensor, batch_normalization=True, scope=None, reuse=True):
        """[summary]
        
        Arguments:
            input_tensor {[type]} -- [description]
        
        Keyword Arguments:
            batch_normalization {bool} -- [description] (default: {True})
            scope {[type]} -- [description] (default: {None})
            reuse {bool} -- [description] (default: {True})
        """
        with tf.variable_scope(scope, 'BlockInceptionA', [input_tensor], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                avg_pool_1 = avarage_pool_2d(input_tensor=input_tensor, kernel_sizes=(3,3), padding='SAME', stride_sizes=(1,1), names='0a_3x3')
                conv_1 = self.conv_block(inputs=avg_pool_1, filters=96, kernel_size=(1,1), padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='1a_1x1')
            with tf.variable_scope('Branch_1'):
                conv_2 = self.conv_block(inputs=input_tensor, filters=96, kernel_size=(1,1), padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='1b_1x1')
            with tf.variable_scope('Branch_2'):
                conv_4 = self.conv_block(inputs=input_tensor, filters=64, kernel_size=(1,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='0c_1x1')
                conv_5 = self.conv_block(inputs=conv_4, filters=96, kernel_size=(3,3) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='1c_3x3')
            with tf.variable_scope('Branch_3'):
                conv_6 = self.conv_block(inputs=input_tensor, filters=64, kernel_size=(1,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='0d_1x1')
                conv_7 = self.conv_block(inputs=conv_6, filters=96, kernel_size=(3,3) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='1d_3x3')
                conv_8 = self.conv_block(inputs=conv_7, filters=96, kernel_size=(3,3) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='2d_3x3')

            concat_1 = concatenate(list_tensor=[conv_1,conv_2,conv_5,conv_8], axis=-1, names=None)
        return concat_1    
    
    # Reduction A
    def reduction_a(self, input_tensor, batch_normalization=True, scope=None, reuse=None):
        """[summary]
        
        Arguments:
            input_tensor {[type]} -- [description]
        
        Keyword Arguments:
            batch_normalization {bool} -- [description] (default: {True})
            scope {[type]} -- [description] (default: {None})
            reuse {[type]} -- [description] (default: {None})
        """
        with tf.variable_scope(scope, 'BlockReductionA', [input_tensor], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                max_pool_1 = max_pool_2d(input_tensor=input_tensor, pool_sizes=(3,3), stride_sizes=(2,2), paddings='VALID', names='0a_3x3')
            with tf.variable_scope('Branch_1'):
                conv_1 = self.conv_block(inputs=input_tensor, filters=384, kernel_size=(3,3) , strides=(2,2), use_batch_norm=batch_normalization, name='0b_3x3')
            with tf.variable_scope('Branch_2'):
                conv_2 = self.conv_block(inputs=input_tensor, filters=192, kernel_size=(1,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='0c_1x1')
                conv_3 = self.conv_block(inputs=conv_2, filters=224, kernel_size=(3,3) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='1c_3x3')
                conv_4 = self.conv_block(inputs=conv_3, filters=256, kernel_size=(3,3) , strides=(2,2), use_batch_norm=batch_normalization, name='0d_1x1')

            concat_1 = concatenate(list_tensor=[max_pool_1,conv_1,conv_4], axis=-1, names=None)
        return concat_1
    
    
    # Inception B
    def inception_b(self, input_tensor, batch_normalization=True, scope=None, reuse=None):
        """[summary]
        
        Arguments:
            input_tensor {[type]} -- [description]
        
        Keyword Arguments:
            batch_normalization {bool} -- [description] (default: {True})
            scope {[type]} -- [description] (default: {None})
            reuse {[type]} -- [description] (default: {None})
        """
        with tf.variable_scope(scope, 'BlockInceptionB', [input_tensor], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                avg_pool_1 = avarage_pool_2d(input_tensor=input_tensor, kernel_sizes=(3,3), padding='SAME', stride_sizes=(1,1), names='0a_3x3')
                conv_1 = self.conv_block(inputs=avg_pool_1, filters=128, kernel_size=(1,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='1a_1x1')
            with tf.variable_scope('Branch_1'):
                conv_2 = self.conv_block(inputs=input_tensor, filters=384, kernel_size=(1,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='0b_1x1')
            with tf.variable_scope('Branch_2'):
                conv_3 = self.conv_block(inputs=input_tensor, filters=192, kernel_size=(1,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='0c_1x1')
                conv_4 = self.conv_block(inputs=conv_3, filters=224, kernel_size=(1,7) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='1c_1x7')
                conv_5 = self.conv_block(inputs=conv_4, filters=256, kernel_size=(1,7) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='2c_1x7')
            with tf.variable_scope('Branch_3'):
                conv_6 = self.conv_block(inputs=input_tensor, filters=192, kernel_size=(1,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='0d_1x1')
                conv_7 = self.conv_block(inputs=conv_6, filters=192, kernel_size=(1,7) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='1d_1x7')
                conv_8 = self.conv_block(inputs=conv_7, filters=224, kernel_size=(7,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='2d_7x1')
                conv_9 = self.conv_block(inputs=conv_8, filters=224, kernel_size=(1,7) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='3d_1x7')
                conv_10 = self.conv_block(inputs=conv_9, filters=256, kernel_size=(7,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='4d_7x1')

            concat_1 = concatenate(list_tensor=[conv_1,conv_2,conv_5,conv_10], axis=-1, names=None)
        return concat_1

    # Reduction B
    def reduction_b(self, input_tensor, batch_normalization=True, scope=None, reuse=None):
        """[summary]
        
        Arguments:
            input_tensor {[type]} -- [description]
        
        Keyword Arguments:
            batch_normalization {bool} -- [description] (default: {True})
            scope {[type]} -- [description] (default: {None})
            reuse {[type]} -- [description] (default: {None})
        """
        with tf.variable_scope(scope, 'BlockReductionB', [input_tensor], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                max_pool_1 = max_pool_2d(input_tensor=input_tensor, pool_sizes=(3,3), stride_sizes=(2,2), paddings='VALID', names='0a_3x3')
            with tf.variable_scope('Branch_1'):
                conv_1 = self.conv_block(inputs=input_tensor, filters=192, kernel_size=(1,1) , strides=(1,1), padding='SAME', use_batch_norm=batch_normalization, name='0b_1x1')
                conv_2 = self.conv_block(inputs=conv_1, filters=192, kernel_size=(1,1) , strides=(2,2), padding='VALID', use_batch_norm=batch_normalization, name='0b_1x1')
            with tf.variable_scope('Branch_2'):
                conv_3 = self.conv_block(inputs=input_tensor, filters=256, kernel_size=(1,1) , strides=(1,1), padding='SAME', use_batch_norm=batch_normalization, name='0c_1x1')
                conv_4 = self.conv_block(inputs=conv_3, filters=256, kernel_size=(1,7) , strides=(1,1), padding='SAME', use_batch_norm=batch_normalization, name='1c_1x7')
                conv_5 = self.conv_block(inputs=conv_4, filters=320, kernel_size=(7,1) , strides=(1,1), padding='SAME', use_batch_norm=batch_normalization, name='2c_7x1')
                conv_6 = self.conv_block(inputs=conv_5, filters=320, kernel_size=(3,3) , strides=(2,2), padding='VALID', use_batch_norm=batch_normalization, name='3c_3x3')
            
            concat_1 = concatenate(list_tensor=[max_pool_1,conv_2,conv_6], axis=-1, names=None)
        return concat_1

    # Inception C
    def inception_c(self, input_tensor, batch_normalization=True, scope=None, reuse=None):
        """[summary]
        
        Arguments:
            input_tensor {[type]} -- [description]
        
        Keyword Arguments:
            batch_normalization {bool} -- [description] (default: {True})
            scope {[type]} -- [description] (default: {None})
            reuse {[type]} -- [description] (default: {None})
        """
        with tf.variable_scope(scope, 'BlockInceptionC', [input_tensor], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                avg_pool_1 = avarage_pool_2d(input_tensor=input_tensor, kernel_sizes=(3,3), padding='SAME', stride_sizes=(1,1), names='0a_3x3')
                conv_1 = self.conv_block(inputs=avg_pool_1, filters=256, kernel_size=(1,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='1a_1x1')
            with tf.variable_scope('Branch_1'):
                conv_2 = self.conv_block(inputs=input_tensor, filters=256, kernel_size=(1,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='0b_1x1')
            with tf.variable_scope('Branch_2'):
                conv_3 = self.conv_block(inputs=input_tensor, filters=384, kernel_size=(1,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='0c_1x1')
                with tf.variable_scope('Branch_21'):
                    conv_4 = self.conv_block(inputs=conv_3, filters=256, kernel_size=(1,3) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='1c1_1x3')
                with tf.variable_scope('Branch_22'):
                    conv_5 = self.conv_block(inputs=conv_3, filters=256, kernel_size=(3,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='1c2_3x1')
            with tf.variable_scope('Branch_3'):
                conv_6 = self.conv_block(inputs=input_tensor, filters=384, kernel_size=(1,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='0d_1x1')
                conv_7 = self.conv_block(inputs=conv_6, filters=448, kernel_size=(1,3) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='1d_1x3')
                conv_8 = self.conv_block(inputs=conv_7, filters=512, kernel_size=(3,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='2d_3x1')
                with tf.variable_scope('Branch_31'):
                    conv_9 = self.conv_block(inputs=conv_8, filters=256, kernel_size=(3,1) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='3d1_3x1')
                with tf.variable_scope('Branch_32'):
                    conv_10 = self.conv_block(inputs=conv_8, filters=256, kernel_size=(1,3) , padding='SAME', strides=(1,1), use_batch_norm=batch_normalization, name='3d2_1x3')

            concat_1 = concatenate(list_tensor=[conv_1,conv_2,conv_4,conv_5,conv_9,conv_10], axis=-1, names=None)
        return concat_1

    # Create Model
    def create_model(self, input_tensor, classes=1000, n_inception_a=4, n_inception_b=7, n_inception_c=3, batch_normalizations=True, scope=None):
        """[summary]
        
        Arguments:
            input_tensor {[type]} -- [description]
        
        Keyword Arguments:
            classes {int} -- [description] (default: {1000})
            n_inception_a {int} -- [description] (default: {4})
            n_inception_b {int} -- [description] (default: {7})
            n_inception_c {int} -- [description] (default: {3})
            batch_normalizations {bool} -- [description] (default: {True})
        """
        with tf.variable_scope(scope, 'inception_v4', [input_tensor]):
            stem_layer = self.stem_block(input_tensor=input_tensor, batch_normalization=batch_normalizations)

            for i in range(0,n_inception_a):
                inception_a_layer = self.inception_a(input_tensor=stem_layer, batch_normalization=batch_normalizations)

            reduction_a_layer = self.reduction_a(input_tensor=inception_a_layer, batch_normalization=batch_normalizations)

            for j in range(0,n_inception_b):
                inception_b_layer = self.inception_b(input_tensor=reduction_a_layer, batch_normalization=batch_normalizations)

            reduction_b_layer = self.reduction_b(input_tensor=inception_b_layer, batch_normalization=batch_normalizations)

            for k in range(0,n_inception_c):
                inception_c_layer = self.inception_c(input_tensor=reduction_b_layer, batch_normalization=batch_normalizations)

            if n_inception_a == 0:
                inception_v4 = stem_layer
            elif n_inception_b == 0:
                inception_v4 = reduction_a_layer
            elif n_inception_c == 0:
                inception_v4 = reduction_b_layer
            else:
                inception_v4 = inception_c_layer

            base_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            inception_v4 = flatten(input_tensor=inception_v4, names='output')

            inception_v4 = avarage_pool_2d(input_tensor=inception_v4, kernel_sizes=(3,3), padding='SAME', stride_sizes=(1,1), names='output')

            inception_v4 =  dropout(input_tensor=inception_v4, names='output', dropout_rates=0.2)

            non_logit = dense(input_tensor=inception_v4, hidden_units=classes, names='output')
            
            if classes > 2:
                output = softmax(input_tensor=non_logit)
            else:
                output = sigmoid(input_tensor=non_logit)

            full_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return non_logit, output, base_var_list, full_var_list

