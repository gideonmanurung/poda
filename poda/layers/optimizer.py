import tensorflow as tf

def optimizers(optimizers_name='adam',learning_rate=0.0001):
    optimizers_function = None
    if optimizers_name=='adam':
        optimizers_function = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')
    elif optimizers_name=='adadelta':
        optimizers_function = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,rho=0.95,epsilon=1e-08,use_locking=False,name='Adadelta')
    elif optimizers_name=='adagrad':
        optimizers_function = tf.train.AdagradOptimizer(learning_rate=learning_rate,initial_accumulator_value=0.1,use_locking=False,name='Adagrad')
    elif optimizers_name=='gradient':
        optimizers_function = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,use_locking=False,name='GradientDescent')
    elif optimizers_name=='rmsprop':
        optimizers_function = tf.train.RMSPropOptimizer(learning_rate=learning_rate,decay=0.9,momentum=0.0,epsilon=1e-10,use_locking=False,centered=False,name='RMSProp')
    return optimizers_function