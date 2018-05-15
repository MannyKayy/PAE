import tensorflow as tf
import numpy as np

def linear(input, w, b):
    '''
        A linear computation
    '''
    return tf.add(tf.matmul(w,input), b)

def get_layer(input, dim_out, dim_in, activ, batch_norm=False, train=True, scope=None, reuse=None):
    '''
        Return output of a layer
    '''
    input = tf.transpose(input) # to agree with tensorflow convention [batch_size, dim]
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
    with tf.variable_scope(scope, reuse=reuse):
        l = tf.contrib.layers.fully_connected(input, dim_out, activation_fn=None, weights_regularizer=regularizer, reuse=reuse, scope='dense')
        if batch_norm:
            logits = tf.contrib.layers.batch_norm(l, center=True, scale=True, is_training=train, reuse=reuse, scope='bn')
        else:
            logits = l
        if activ == 'relu':
            pred = tf.nn.relu(logits)
        elif activ == 'sigmoid':
            pred = tf.sigmoid(logits)
        elif activ == 'linear':
            pred = logits
        elif activ == 'softplus':
            pred = tf.nn.softplus(logits)
        elif activ == 'selu':
            pred = tf.nn.selu(logits)
        elif activ == 'elu':
            pred = tf.nn.elu(logits)
        elif activ == 'tanh':
            pred = tf.tanh(logits)
        elif activ == 'softmax':
            pred = tf.nn.softmax(logits)
        return tf.transpose(pred) # transpose back to our convention