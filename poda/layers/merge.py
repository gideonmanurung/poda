import tensorflow as tf

def add(input_tensor_1, input_tensor_2, names=None):
    """[summary]
    
    Arguments:
        input_tensor_1 {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer 1]
        input_tensor_2 {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer 2]
        names {[str]} -- [Name of the layer] (default: {None})

    Returns:
        [Tensor] -- [Added tensor]
    """
    if names!=None:
        names = str(names)
    else:
        names = ''

    add_tensor = tf.math.add(x=input_tensor_1,y=input_tensor_2,name=names)
    return add_tensor

def concatenate(input_tensor_1, input_tensor_2, axis=-1, names=None):
    """[summary]
    
    Arguments:
        input_tensor_1 {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer 1]
        input_tensor_2 {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer 2]
    
    Keyword Arguments:
        axis {int} -- [Dimension tensor] (default: {-1})
        names {[str]} -- [Name of the layer] (default: {None})
    """
    if names!=None:
        names = str(names)
    else:
        names = ''

    concat_tensor = tf.concat(values=[input_tensor_1,input_tensor_2],axis=axis,name=names)
    return concat_tensor

def maximum(input_tensor_1, input_tensor_2, names=None):
    """[summary]
    
    Arguments:
        input_tensor_1 {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer 1]
        input_tensor_2 {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer 2]
        names {[str]} -- [Name of the layer] (default: {None})

    Returns:
        [Tensor] -- [maximum tensor]
    """
    if names!=None:
        names = str(names)
    else:
        names = ''

    maximum_tensor = tf.math.maximum(x=input_tensor_1,y=input_tensor_2,name=names)
    return maximum_tensor

def minimum(input_tensor_1, input_tensor_2, names=None):
    """[summary]
    
    Arguments:
        input_tensor_1 {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer 1]
        input_tensor_2 {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer 2]
        names {[str]} -- [Name of the layer] (default: {None})

    Returns:
        [Tensor] -- [minimum tensor]
    """
    if names!=None:
        names = str(names)
    else:
        names = ''

    minimum_tensor = tf.math.minimum(x=input_tensor_1,y=input_tensor_2,name=names)
    return minimum_tensor

def multiply(input_tensor_1, input_tensor_2, names=None):
    """[summary]
    
    Arguments:
        input_tensor_1 {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer 1]
        input_tensor_2 {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer 2]
        names {[str]} -- [Name of the layer] (default: {None})

    Returns:
        [Tensor] -- [Multiplied tensor]
    """
    if names!=None:
        names = str(names)
    else:
        names = ''

    multiply_tensor = tf.math.multiply(x=input_tensor_1,y=input_tensor_2,name=names)
    return multiply_tensor

def substract(input_tensor_1, input_tensor_2, names=None):
    """[summary]
    
    Arguments:
        input_tensor_1 {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer 1]
        input_tensor_2 {[float, double, int32, int64, uint8, int16, or int8]} -- [A Tensor representing prelayer 2]
        names {[str]} -- [Name of the layer] (default: {None})

    Returns:
        [Tensor] -- [Substracted tensor]
    """
    if names!=None:
        names = str(names)
    else:
        names = ''

    substract_tensor = tf.math.subtract(x=input_tensor_1,y=input_tensor_2,name=names)
    return substract_tensor

