import tensorflow as tf
from poda.layers.activation import *
from poda.layers.regularizer import *

def new_weights(shapes, names, random_types='truncated_normal', dtypes=tf.float32):
    """[Create a new trainable tensor as weight]
    
    Arguments:
        shapes {[Array of int]} -- [- example (convolution), [filter height, filter width, input channels, output channels]
                                    - example (fully connected), [num input, num output]]
        names {[str]} -- [A name for the operation (optional)]
    
    Keyword Arguments:
        random_types {str} -- [Type of random values] (default: {'truncated_normal'})
        dtypes {[float32]} -- [Data type of tensor] (default: {tf.float32})
    
    Returns:
        [float32] -- [A trainable tensor]
    """
    if random_types=='random_normal':
        initial_weight = tf.random_normal(shape=shapes,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None)
    elif random_types=='random_uniform':
        initial_weight = tf.random_uniform(shape=shapes,minval=0,maxval=None,dtype=tf.float32,seed=None)
    else:
        initial_weight = tf.truncated_normal(shape=shapes,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None)
    return tf.Variable(initial_weight, dtype=dtypes, name='weight_'+names)

def new_biases(shapes, names, dtypes=tf.float32):
    """[Create a new trainable tensor as bias]
    
    Arguments:
        shapes {[int]} -- [Length of filter or node ]
        names {[str]} -- [A name for the operation (optional)]
    
    Keyword Arguments:
        dtypes {[float32]} -- [Data type of tensor] (default: {tf.float32})
    
    Returns:
        [float32] -- [A trainable tensor]
    """
    return tf.Variable(tf.constant(0.05, shape=[shapes], dtype=dtypes), dtype=dtypes, name='bias_'+names)

def fully_connected(input_tensor, hidden_units, names, activation=None, regularizers=None, scale=0.2):
    """[Create a new trainable Fully Connected layer]
    
    Arguments:
        input_tensor {[float, double, int32, int64, uint8, int16, or int8]} -- [description]
        hidden_units {[int]} -- [description]
        names {[type]} -- [description]
    
    Keyword Arguments:
        activation {str} -- [description] (default: {'relu'})
        regularizers {[type]} -- [description] (default: {None})
        scale {float} -- [description] (default: {0.2})
    
    Returns:
        [Tensor] -- [A trainable tensor]
    """
    #num_input , num_output = input_tensor.shape
    #weight = new_weights(shapes=[num_output, hidden_units], names=names)
    #bias = new_biases(shapes=hidden_units, names=names)

    #layer = tf.matmul(input_tensor, weight)
    #layer += bias

    layer = tf.layers.dense(inputs=input_tensor,units=hidden_units,activation=None,
                                use_bias=True,
                                kernel_initializer=None,
                                bias_initializer=tf.zeros_initializer(),
                                kernel_regularizer=None,
                                bias_regularizer=None,
                                activity_regularizer=None,
                                kernel_constraint=None,
                                bias_constraint=None,
                                trainable=True,
                                name=names,
                                reuse=None)

    if activation!=None:
        layer = define_activation_function(input_tensor=layer, activation_name=activation, names=names)
    else:
        layer = layer

    if regularizers=='both':
        layer = dropout(input_tensor=layer, dropout_rates=scale, names=names)
        layer = l2_regularization(input_tensor=layer, names=names)
    elif regularizers=='l1':
        layer = l1_regularization(input_tensor=layer)
    elif regularizers=='l2':
        layer = l2_regularization(input_tensor=layer)
    elif regularizers=='dropout':
        layer = dropout(input_tensor=layer, dropout_rates=scale, names=names)
    else:
        layer = layer

    return layer
    
