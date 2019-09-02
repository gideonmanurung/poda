import tensorflow as tf

def optimizers(optimizers_names='adam',learning_rates=0.0001):
    optimizers_function = None
    if activation_names=='adam':
        optimizers_function = tf.train.AdamOptimizer(learning_rate=learning_rates,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')
    elif activation_names=='adadelta':
        optimizers_function = tf.train.AdadeltaOptimizer(learning_rate=learning_rates,rho=0.95,epsilon=1e-08,use_locking=False,name='Adadelta')
    elif activation_names=='adagrad':
        optimizers_function = tf.train.AdagradOptimizer(learning_rate=learning_rates,initial_accumulator_value=0.1,use_locking=False,name='Adagrad')
    elif activation_names=='gradient':
        optimizers_function = tf.train.GradientDescentOptimizer(learning_rate=learning_rates,use_locking=False,name='GradientDescent')
    elif activation_names=='rmsprop':
        optimizers_function = tf.train.RMSPropOptimizer(learning_rate=learning_rates,decay=0.9,momentum=0.0,epsilon=1e-10,use_locking=False,centered=False,name='RMSProp')
    elif optimizers_names=='sgd':
        optimizers_function = tf.train.GradientDescentOptimizer(learning_rate=learning_rates)
    return optimizers_function