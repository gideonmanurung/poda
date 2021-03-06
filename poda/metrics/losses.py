import tensorflow as tf


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

    
