from __future__ import division
import argparse
import os
import tensorflow as tf 
from tqdm import tqdm
import numpy as np
from discriminator import Discriminator
from data import *
import matplotlib
import matplotlib.pyplot as plt

ARCH_MLP = [1000, 256, 128, 10]

def load_model(working_dir, sess):
    saver = tf.train.Saver()
    path = os.path.join(working_dir, 'model')
    ckpt = tf.train.latest_checkpoint(path)
    print('\nFound latest model: %s'%ckpt)
    if ckpt:
        saver.restore(sess, ckpt)
        print('\nLoaded %s\n'%ckpt)

def save_model(working_dir, sess):
    saver = tf.train.Saver()
    path = os.path.join(working_dir, 'model/saved-model')
    save_path = saver.save(sess, path)

class CNN_CLF:
    def __init__(self, data_dist, img_w, img_h, img_c, n_classes):
        self.data_dist = data_dist
        self.img_w = img_w
        self.img_h = img_h
        self.img_c = img_c
        self.n_classes = n_classes
        self.learning_rate = 1e-3
        self.max_iters = 1e5
        self.skip = 100
        self.batch_size = 128

    def reshape_data(self, x, db):
        N = x.shape[1]
        w = self.img_w
        h = self.img_h
        channels = self.img_c
        if db == 'color_mnist':
            x_new = np.zeros((N,self.img_w,self.img_h,self.img_c))
            for i in range(N):
                for c in range(channels):
                    x_new[i,:,:,c] = x[c*w*h:(c+1)*w*h,i].reshape(w,h)
            return x_new
        else: # do nothing
            return x
    
    def create_model(self):
        w = self.img_w
        h = self.img_h
        channels = self.img_c
        self.x = tf.placeholder(tf.float32, shape=(None,w,h,channels))
        self.y = tf.placeholder(tf.int32, shape=None)
        # Convolutional Layer #1 and Pooling Layer #1
        conv1 = tf.layers.conv2d(
                inputs=self.x,
                filters=6,
                kernel_size=[5, 5],
                activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)  
        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=16,
                kernel_size=[5, 5],
                activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, int(w/4-3)*int(h/4-3)*16])
        dense = tf.layers.dense(inputs=pool2_flat, units=120, activation=tf.nn.relu)
        dense = tf.layers.dense(inputs=dense, units=84, activation=tf.nn.relu)
        # Logits Layer
        self.logits = tf.layers.dense(inputs=dense, units=self.n_classes)
        _,preds = tf.nn.top_k(tf.nn.softmax(self.logits))
        self.preds = tf.reshape(preds,shape=[-1])
        
        # Loss
        onehot_labels = tf.one_hot(self.y, depth=self.n_classes, on_value=1.0,off_value=0.0)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=self.logits)
        
        # optimizer
        self.all_params = tf.trainable_variables()
        self.global_step = tf.Variable(0, trainable=False)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.loss,
                var_list=self.all_params,
                global_step=self.global_step
        )

    def predict(self,x,sess=None):
        if sess is None:
            sess = tf.get_default_session()
        return sess.run(self.preds,feed_dict={
            self.x:x
        })

    def train(self):
        init = tf.global_variables_initializer()
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            load_model(args.working_dir, sess)
            n_iters_left = self.max_iters - self.global_step.eval()
            t_obj = tqdm(range(int(n_iters_left)))
            for t in t_obj:
                x,y = self.data_dist.sample(self.batch_size)
                x = self.reshape_data(x, args.db)
                
                _,loss = sess.run([self.opt,self.loss], feed_dict={
                    self.x:x,
                    self.y:y
                })
                val_x, val_y = self.data_dist.val_sample(self.batch_size)
                val_x = self.reshape_data(val_x, args.db)
                preds = self.predict(val_x)
                                
                acc = np.sum(preds == val_y)/val_y.shape[0]
                t_obj.set_description('loss: %.3f, validation acc: %.3f'%(loss,acc))
                if self.global_step.eval() % self.skip == 0:
                    save_model(args.working_dir, sess)


class MLP_CLF:
    def __init__(self, data_dist, arch):
        self.data_dist = data_dist
        self.data_dim = arch[0]
        self.n_classes = arch[-1]
        self.disc_arch = arch
        self.learning_rate = 1e-3
        self.max_iters = 1e3
        self.skip = 10
        self.batch_size = 128

    def create_model(self):
        self.x = tf.placeholder(tf.float32, shape=(self.data_dim, None))
        self.y = tf.placeholder(tf.int32, shape=None)

        classifier = Discriminator(self.disc_arch, 'relu', 'linear', 'DISC')
        self.logits = tf.transpose(classifier(self.x)) # [batch_size, num_classes]
        _,preds = tf.nn.top_k(tf.nn.softmax(self.logits))
        self.preds = tf.reshape(preds,shape=[-1])
        onehot_labels = tf.one_hot(self.y,depth=self.n_classes,on_value=1.0,off_value=0.0)
        
        # loss
        self.loss = tf.losses.softmax_cross_entropy(logits=self.logits, \
                    onehot_labels=onehot_labels)
        
        # optimizer
        self.all_params = tf.trainable_variables()
        self.global_step = tf.Variable(0, trainable=False)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.loss,
                var_list=self.all_params,
                global_step=self.global_step
        )

    def predict(self,x,sess=None):
        if sess is None:
            sess = tf.get_default_session()
        return sess.run(self.preds,feed_dict={
            self.x:x
        })

    def train(self):
        init = tf.global_variables_initializer()
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            load_model(args.working_dir, sess)
            n_iters_left = self.max_iters - self.global_step.eval()
            t_obj = tqdm(range(int(n_iters_left)))
            for t in t_obj:
                x,y = self.data_dist.sample(self.batch_size)
                _,loss = sess.run([self.opt,self.loss], feed_dict={
                    self.x:x,
                    self.y:y
                })
                val_x, val_y = self.data_dist.val_sample(self.batch_size)
                preds = self.predict(val_x)
                acc = np.sum(preds == val_y)/val_y.shape[0]
                t_obj.set_description('loss: %f, validation acc: %f'%(loss,acc))
                if self.global_step.eval() % self.skip == 0:
                    save_model(args.working_dir, sess)
                

parser = argparse.ArgumentParser(description='Train classifiers')
parser.add_argument('--db', action='store', required=True, help='one of datasets: \
                                                        low_dim_embed (10d gaussians embedded in 1000d space)\
                                                        color_mnist (stacked MNIST),\
                                                        cifar_100 (CIFAR 100 categories)')
parser.add_argument('--working_dir', action='store', required=True, help='path to save and load model information')


'''
main function
'''
if __name__ == '__main__':
    args = parser.parse_args()
    WORKING_DIR = args.working_dir
    if not os.path.isdir('classification'):
        os.mkdir('classification')
    if not os.path.isdir(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    if not os.path.isdir(os.path.join(WORKING_DIR,'model')):
        os.mkdir(os.path.join(WORKING_DIR,'model'))

    if args.db == 'low_dim_embed':
        data_dist = LowDimEmbed()
        clf = MLP_CLF(data_dist, ARCH_MLP)
    elif args.db == 'color_mnist':
        data_dist = CMNIST('data/mnist')
        clf = CNN_CLF(data_dist, 28, 28, 3, 1000)

    clf.create_model()
    clf.train()
