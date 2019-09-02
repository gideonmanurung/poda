import tensorflow as tf 
import numpy as np


def calculate_accuracy(input_tensor, label, threshold=0.5):
    """[summary]
    
    Arguments:
        input_tensor {[type]} -- [description]
        label {[type]} -- [description]
    
    Keyword Arguments:
        threshold {float} -- [description] (default: {0.5})
    
    Returns:
        [type] -- [description]
    """
    mask = tf.fill(tf.shape(label), 1.)
    input_tensor = tf.reshape(input_tensor, [tf.shape(input_tensor)[0], -1])
    label = tf.reshape(label, [tf.shape(label)[0], -1])

    input_tensor = tf.math.greater(input_tensor, tf.convert_to_tensor(np.array(threshold), tf.float32))
    input_tensor = tf.cast(input_tensor, tf.float32)
    label = tf.math.greater(label, tf.convert_to_tensor(np.array(threshold), tf.float32))
    label = tf.cast(label, tf.float32)

    error = tf.reduce_sum(tf.abs(input_tensor - label)) / (tf.reduce_sum(mask) + 0.0001)
    acc = 1. - error
    return acc

def mse_loss_mean(output_tensor, label):
    """"[summary]
    
    Arguments:
        output_tensor {tensor} -- [description]
        label {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    loss = tf.square(tf.subtract(output_tensor, label))
    loss = tf.reduce_mean(loss)
    return loss


def mse_loss_sum(output_tensor, label):
    """"[summary]
    
    Arguments:
        output_tensor {tensor} -- [description]
        label {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    loss = tf.square(tf.subtract(output_tensor, label))
    loss = tf.reduce_sum(loss)
    return loss


def softmax_crosentropy_mean(output_tensor, label):
    """[summary]
    
    Arguments:
        output_tensor {[type]} -- [description]
        label {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=output_tensor)
    loss = tf.reduce_mean(loss)
    return loss


def softmax_crosentropy_sum(output_tensor, label):
    """[summary]
    
    Arguments:
        output_tensor {[type]} -- [description]
        label {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=output_tensor)
    loss = tf.reduce_sum(loss)
    return loss


def sigmoid_crossentropy_mean(output_tensor, label):
    """[summary]
    
    Arguments:
        output_tensor {[type]} -- [description]
        label {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=output_tensor)
    loss = tf.reduce_mean(loss)
    return loss


def sigmoid_crossentropy_sum(output_tensor, label):
    """[summary]
    
    Arguments:
        output_tensor {[type]} -- [description]
        label {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=output_tensor)
    loss = tf.reduce_sum(loss)
    return loss