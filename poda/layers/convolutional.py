import tensorflow as tf
from poda.layers.activation_function import *
from poda.layers.dense import *
from poda.layer.regularizer import *

def convolution_1d(input_tensor, number_filters, kernel_sizes=3, stride_sizes=(1,1), padding='valid', activation='relu', dropout_layers=None, names=None):
    """[summary]
    
    Arguments:
        input_tensor {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer values]
        number_filters {[int]} -- [the dimensionality of the output space (i.e. the number of filters in the convolution).]
    
    Keyword Arguments:
        kernel_sizes {int} -- [description] (default: {3})
        stride_sizes {tuple} -- [description] (default: {(1,1)})
        padding {str} -- [description] (default: {'valid'})
        activation {str} -- [description] (default: {'relu'})
        dropout_layers {[type]} -- [description] (default: {None})
        name {[str]} -- [Name of the layer] (default: {None})
    
    Returns:
        [Tensor] -- [Layer convolution 1D with dtype tf.float32]
    """
    weight = new_weights(shapes=[kernel_sizes, input_tensor[2], number_filters], names=names)
    layer = tf.nn.conv1d(value=input_tensor, filters=weight, stride=stride_sizes[0], padding=padding, use_cudnn_on_gpu=True, data_format=None, name='conv_1d_'+names)

    layer = define_activation_function(input_tensor=layer, activation_name=activation, names=names)

    if dropout_layer!=None:
        layer = dropout(input_tensor=layer, names=names)
    else:
        layer = layer
    return layer

def convolution_2d(input_tensor, number_filters, kernel_sizes=(3,3), stride_sizes=(1,1), padding='valid', activation='relu', dropout_layer=None, names=None):
    """[This function creates a convolution kernel that is convolved (actually cross-correlated) with the layer input to produce a tensor of outputs]
    
    Arguments:
        input_tensor {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer values]
        number_filters {[int]} -- [the dimensionality of the output space (i.e. the number of filters in the convolution).]
    
    Keyword Arguments:
        kernel_sizes {int} -- [description] (default: {(3,3)})
        stride_sizes {tuple} -- [description] (default: {(1,1)})
        padding {str} -- [description] (default: {'valid'})
        activation {str} -- [description] (default: {'relu'})
        dropout_layer {[float]} -- [description] (default: {None})
        name {[str]} -- [Name of the layer] (default: {None})
    
    Returns:
        [Tensor] -- [Layer convolution 2D with dtype tf.float32]
    """
    weight = new_weights(shapes=[kernel_sizes, input_tensor[3], number_filters], names=names)
    layer = tf.nn.conv2d(input=input_tensor, filter=weight, strides=[stride_sizes,stride_sizes], padding=padding, use_cudnn_on_gpu=True,
                         data_format='NHWC', dilations=[1, 1, 1, 1], name='conv_2d_'+names)

    layer = define_activation_function(input_tensor=layer, activation_name=activation, names=names)
    if dropout_layer!=None:
        layer = dropout(input_tensor=layer, names=names)
    else:
        layer = layer
    return layer

def convolution_3d(input_tensor, number_filters, kernel_sizes=(3,3), stride_sizes=(1,1), padding='valid', activation='relu', dropout_layer=None, names=None):
    weight = new_weights(shapes=[kernel_sizes, input_tensor[3], number_filters], names=names)
    layer = tf.nn.conv3d(input,filter,strides=[stride_sizes,stride_sizes,stride_sizes],padding=padding,data_format='NDHWC',dilations=[1, 1, 1, 1, 1],name='conv_3d_'+names)
    return layer

def depthwise_convolution_2d(input_tensor, number_filters, kernel_sizes=(3,3), stride_sizes=(1,1), padding='valid', activation='relu', dropout_layer=None, names=None):
    """[Function for adding depthwise convolution 2D layer]
    
    Arguments:
        input_tensor {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer values]
        number_filters {[int]} -- [the dimensionality of the output space (i.e. the number of filters in the convolution).]
    
    Keyword Arguments:
        kernel_size {int , int} -- [Size of kernel] (default: {(3,3)})
        stride_size {int , int} -- [Size of striding of kernel] (default: {(1,1)})
        padding {str} -- [Type of padding. Ex. same and valid] (default: {'valid'})
        activation_function {str} -- [Type of activation function in layer] (default: {'relu'})
        dropout_layer {[float]} -- [Value of dropout rate and determine to use dropout or not] (default: {None})
        name {[str]} -- [Name of the layer] (default: {None})
    
    Returns:
        [Tensor] -- [Layer depthwise convolution 2D with dtype tf.float32]
    """
    weight = new_weights(shapes=[kernel_sizes, input_tensor[3], number_filters], names=names)
    layer = tf.nn.depthwise_conv2d(input=input_tensor, filter=weight, strides=[stride_sizes,stride_sizes], padding=padding, rate=None, name='deptwise_conv_2d_'+names, data_format=None )

    layer = define_activation_function(input_tensor=layer, activation_name=activation, names=names)
    if dropout_layer!=None:
        layer = dropout(input_tensor=layer, names=names)
    else:
        layer = layer
    return layer