import tensorflow as tf
from poda.layers.activation import *
from poda.layers.dense import *
from poda.layers.regularizer import *

def batch_normalization(input_tensor,is_trainable=True):
    """[summary]
    
    Arguments:
        input_tensor {[type]} -- [A Tensor representing prelayer]
    
    Keyword Arguments:
        is_trainable {bool} -- [description] (default: {True})
    """
    layer = tf.layers.batch_normalization(inputs=input_tensor,axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,beta_initializer=tf.zeros_initializer(),
                                            gamma_initializer=tf.ones_initializer(),moving_mean_initializer=tf.zeros_initializer(),moving_variance_initializer=tf.ones_initializer(),
                                            beta_regularizer=None,gamma_regularizer=None,beta_constraint=None,gamma_constraint=None,training=False,trainable=is_trainable,name=None,reuse=None,
                                            renorm=False,renorm_clipping=None,renorm_momentum=0.99,fused=None,virtual_batch_size=None,adjustment=None)
    return layer

def convolution_1d(input_tensor, number_filters, kernel_sizes=3, stride_sizes=(1,1), paddings='VALID', activations='relu', dropout_layers=None, names=None):
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
    weight = new_weights(shapes=[kernel_sizes, input_tensor.get_shape().as_list()[2], number_filters], names=names)

    if names!=None:
        names = 'conv_2d_'+str(names)
    else:
        names = 'conv_2d'

    layer = tf.nn.conv1d(value=input_tensor, filters=weight, stride=stride_sizes[0], padding=padding, use_cudnn_on_gpu=True, data_format=None, name='conv_1d_'+names)

    if activations!=None:
        layer = define_activation_function(input_tensor=layer, activation_name=activations, names=names)
    else:
        layer = layer

    if dropout_layers!=None:
        layer = dropout(input_tensor=layer, names=names)
    else:
        layer = layer
    return layer

def convolution_2d(input_tensor, number_filters, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='VALID', activations='relu', dropout_layers=None, names=None):
    """[This function creates a convolution kernel that is convolved (actually cross-correlated) with the layer input to produce a tensor of outputs]
    
    Arguments:
        input_tensor {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer values]
        number_filters {[int]} -- [the dimensionality of the output space (i.e. the number of filters in the convolution).]
    
    Keyword Arguments:
        kernel_sizes {int} -- [description] (default: {(3,3)})
        stride_sizes {tuple} -- [description] (default: {(1,1)})
        padding {str} -- [description] (default: {'VALID'})
        activation {str} -- [description] (default: {'relu'})
        dropout_layer {[float]} -- [description] (default: {None})
        name {[str]} -- [Name of the layer] (default: {None})
    
    Returns:
        [Tensor] -- [Layer convolution 2D with dtype tf.float32]
    """
    weight = new_weights(shapes=[kernel_sizes[0], kernel_sizes[1], input_tensor.get_shape().as_list()[3], number_filters], names=names)

    if names!=None:
        names = 'conv_2d_'+str(names)
    else:
        names = 'conv_2d'

    layer = tf.nn.conv2d(input=input_tensor, filter=weight, strides=[stride_sizes[0], stride_sizes[1]], padding=padding, use_cudnn_on_gpu=True,
                         data_format='NHWC', dilations=[1, 1, 1, 1], name='conv_2d_'+names)

    if activations!=None:
        layer = define_activation_function(input_tensor=layer, activation_name=activations, names=names)
    else:
        layer = layer

    if dropout_layers!=None:
        layer = dropout(input_tensor=layer, names=names)
    else:
        layer = layer
    return layer

def convolution_3d(input_tensor, number_filters, kernel_sizes=(3,3,3), stride_sizes=(1,1), paddings='VALID', activations='relu', dropout_layers=None, names=None):
    """[summary]
    
    Arguments:
        input_tensor {[type]} -- [description]
        number_filters {[type]} -- [description]
    
    Keyword Arguments:
        kernel_sizes {tuple} -- [description] (default: {(3,3,3)})
        stride_sizes {tuple} -- [description] (default: {(1,1)})
        paddings {str} -- [description] (default: {'VALID'})
        activations {str} -- [description] (default: {'relu'})
        dropout_layers {[type]} -- [description] (default: {None})
        names {[type]} -- [description] (default: {None})
    """
    weight = new_weights(shapes=[kernel_sizes[0], kernel_sizes[1], kernel_sizes[1], input_tensor.get_shape().as_list()[3], number_filters], names=names)
    
    if names!=None:
        names = 'conv_2d_'+str(names)
    else:
        names = 'conv_2d'

    layer = tf.nn.conv3d(input,filter,strides=[stride_sizes[0],stride_sizes[1],stride_sizes],padding=paddings,data_format='NDHWC',dilations=[1, 1, 1, 1, 1],name='conv_3d_'+names)
    
    if activations!=None:
        layer = define_activation_function(input_tensor=layer, activation_name=activations, names=names)
    else:
        layer = layer
    
    if dropout_layers!=None:
        layer = dropout(input_tensor=layer, names=names)
    else:
        layer = layer

    return layer

def depthwise_convolution_2d(input_tensor, number_filters, kernel_sizes=(3,3), stride_sizes=(1,1), paddings='VALID', activations='relu', dropout_layers=None, names=None):
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
    weight = new_weights(shapes=[kernel_sizes[0], kernel_sizes[1], input_tensor.get_shape().as_list()[3], number_filters], names=names)

    if names!=None:
        names = 'conv_2d_'+str(names)
    else:
        names = 'conv_2d'

    layer = tf.nn.depthwise_conv2d(input=input_tensor, filter=weight, strides=[stride_sizes[0], stride_sizes[1]], padding=paddings, rate=None, name='deptwise_conv_2d_'+names, data_format=None )

    if activations!=None:
        layer = define_activation_function(input_tensor=layer, activation_name=activations, names=names)
    else:
        layer = layer

    if dropout_layers!=None:
        layer = dropout(input_tensor=layer, names=names)
    else:
        layer = layer
    return layer