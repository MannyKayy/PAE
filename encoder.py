import tensorflow as tf
from layers import *

class Encoder:
    def __init__(self, disc_arch, hidden_acti, output_acti, scope_name, batch_norm=False):
        self.input = input
        self.disc_arch = disc_arch
        self.num_layers = len(self.disc_arch) - 1
        self.hidden_acti = hidden_acti
        self.output_acti = output_acti
        self.scope_name = scope_name
        self.batch_norm = batch_norm

    def __call__(self, input, reuse=None, train=True):
        with tf.variable_scope(self.scope_name, reuse):
            for i in range(self.num_layers):
                if i < self.num_layers - 1:
                    input = get_layer(input, self.disc_arch[i+1], self.disc_arch[i], self.hidden_acti, self.batch_norm, train, scope=str(i), reuse=reuse)
                else:
                    input = get_layer(input, self.disc_arch[i+1], self.disc_arch[i], self.output_acti, False, train, scope=str(i), reuse=reuse)
            return input
