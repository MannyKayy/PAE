from __future__ import division
import tensorflow as tf
import numpy as np
import platform
import matplotlib
if 'Linux' in platform.platform():
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from data import *
from setup import *
from encoder import Encoder
from generator import Generator
from discriminator import Discriminator
import time
import pickle
from tqdm import tqdm

PERTURB = 64

# Adversarial Variational Bayes with Adaptive Contrast
class AVB_AC:
    def __init__(self, data_dist, noise_dist, single_noise_dist, flags, args):
        self.epochs = flags.EPOCHS
        self.num_samples = flags.NUM_SAMPLES
        self.batch_size = flags.BATCH_SIZE
        if args.db in ['grid', 'low_dim_embed']:
            self.net_learning_rate = 2e-5
            self.disc_learning_rate = 1e-4
        else:
            self.net_learning_rate = 5e-5
            self.disc_learning_rate = 1e-4
        self.data_dim = flags.DATA_DIM
        self.noise_dim = flags.NOISE_DIM
        self.perturb = PERTURB
        self.gen_arch = flags.GEN_ARCH
        flags.ENC_ARCH[0] = flags.ENC_ARCH[0]+self.perturb
        self.enc_arch = flags.ENC_ARCH
        self.disc_arch = flags.DISC_ARCH
        self.data_dist = data_dist
        self.noise_dist = noise_dist
        self.single_noise_dist = single_noise_dist
        self.db = flags.DATASET
        self.skip = flags.SKIP
        # setup some alternative definitions given by user
        self.hidden_acti = args.hidden_acti
        self.enc_out_acti = args.enc_out_acti
        self.gen_out_acti = args.gen_out_acti
        self.disc_out_acti = args.disc_out_acti
        self.working_dir = args.working_dir

    def create_model(self):
        self.x = tf.placeholder(tf.float32, shape=(self.data_dim, None))
        self.z = tf.placeholder(tf.float32, shape=(self.noise_dim, None))
        self.eta = tf.placeholder(tf.float32, shape=(self.noise_dim, None))
        self.epsilon = tf.placeholder(tf.float32, shape=(self.perturb, None))

        # Generator
        generator = Generator(self.gen_arch, self.hidden_acti, self.gen_out_acti, 'GEN')
        self.fake_data = generator(self.z)

        # Encoder
        encoder = Encoder(self.enc_arch, self.hidden_acti, self.enc_out_acti, 'ENC')
        self.input = tf.concat([self.x, self.epsilon], 0)
        self.true_input = encoder(self.input)
        self.mean, self.std = tf.nn.moments(self.true_input, axes=[1], keep_dims=True)
        self.std = tf.stop_gradient(tf.sqrt(self.std))
        self.mean = tf.stop_gradient(self.mean)

        # Discriminator
        self.true_input_norm = (self.true_input - self.mean)/self.std
        discriminator = Discriminator(self.disc_arch, self.hidden_acti, self.disc_out_acti, 'DISC')
        fake_pair = tf.concat([self.x, self.true_input_norm], 0)
        true_pair = tf.concat([self.x, self.eta], 0)
        self.fake_output = discriminator(fake_pair)
        self.true_output = discriminator(true_pair, reuse=True)

        # Generator/Encoder cost
        diff = self.x - generator(self.true_input, reuse=True)
        temp = diff**2
        dist = tf.reduce_sum(temp, axis=0, keep_dims=True)
        self.reconstruction = tf.reduce_mean(dist)
        self.net_cost = tf.reduce_mean(self.fake_output) + 0.5*self.reconstruction \
                        - 0.5*tf.reduce_mean(tf.reduce_sum(self.true_input_norm**2, axis=0, keep_dims=True)) \
                        + 0.5*tf.reduce_mean(tf.reduce_sum(self.true_input**2, axis=0, keep_dims=True))

        # Discriminator cost
        temp1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_output, labels=tf.ones_like(self.fake_output))
        temp2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.true_output, labels=tf.zeros_like(self.true_output))
        self.disc_cost = tf.reduce_mean(temp1) + tf.reduce_mean(temp2)

        self.disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DISC')
        self.enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'ENC')
        self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'GEN')
        
        self.disc_global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)

        self.disc_opt = tf.train.AdamOptimizer(self.disc_learning_rate).minimize(
            self.disc_cost,
            var_list=self.disc_params,		
            global_step=self.disc_global_step
        )
        self.net_opt = tf.train.AdamOptimizer(self.net_learning_rate).minimize(
            self.net_cost,
            var_list=self.gen_params+self.enc_params,		
            global_step=self.global_step
        )

    def decode(self, z):
        return self.fake_data.eval(feed_dict={
            self.z:z
        })
        
    def encode(self, x):
        return self.true_input.eval(feed_dict={
            self.x:x,
            self.epsilon:self.single_noise_dist.sample(x.shape[1])
        })

    def load_model(self, sess, ckpt_id=None):
        saver = tf.train.Saver()
        path = os.path.join(self.working_dir, 'model')
        if ckpt_id:
            ckpt = os.path.join(path,'saved-model-' + str(ckpt_id))
            saver.restore(sess, ckpt)
            print('\nLoaded %s\n'%ckpt)
        else:
            ckpt = tf.train.latest_checkpoint(path)
            print('\nFound latest model: %s'%ckpt)
            if ckpt:
                saver.restore(sess, ckpt)
                print('\nLoaded %s\n'%ckpt)

    def save_model(self, sess):
        saver = tf.train.Saver()
        path = os.path.join(self.working_dir, 'model','saved-model')
        save_path = saver.save(sess, path, global_step=self.global_step.eval()+1)
        print('\nModel saved in %s'%save_path)
    
    def save_log(self,log):
        path = os.path.join(self.working_dir,'model','log.pkl')
        with open(path,'wb') as f:
            pickle.dump(log,f)

    def load_log(self,log):
        path = os.path.join(self.working_dir,'model','log.pkl')
        if os.path.exists(path):
            with open(path,'rb') as f:
                data = pickle.load(f)
                log['net_costs'] = data['net_costs']
                log['disc_costs'] = data['disc_costs']
                log['train_time'] = data['train_time']

    def train(self):
        true_data,_ = self.data_dist.sample(self.batch_size)

        init = tf.global_variables_initializer()
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True

        log = {'net_costs':[],'disc_costs':[],'train_time':0}
        start = time.time()
        with tf.Session(config=config) as sess:
            sess.run(init)
            # model loading 
            self.load_model(sess)
            self.load_log(log)
            # how many iters?
            n_iters = int(self.num_samples/self.batch_size)*self.epochs
            n_iters_left = n_iters - self.global_step.eval()
            print('\nNumber of train iterations %d'%n_iters_left)

            for t in tqdm(range(n_iters_left)):
                noise = self.noise_dist.sample(self.batch_size)
                eta = self.noise_dist.sample(self.batch_size)
                epsilon = self.single_noise_dist.sample(self.batch_size)
                _, _ = sess.run([self.disc_opt, self.net_opt], feed_dict={
                    self.z: noise,
                    self.x: true_data,
                    self.epsilon: epsilon,
                    self.eta: eta
                })              
                disc_cost, net_cost = sess.run([self.disc_cost,self.net_cost], feed_dict={
                    self.z: noise,
                    self.x: true_data,
                    self.epsilon: epsilon,
                    self.eta: eta
                })
                if self.global_step.eval() % 100 == 0: # saving at every iteration is not neccessary
                    log['net_costs'].append(net_cost)
                    log['disc_costs'].append(disc_cost)
                    log['train_time'] += (time.time()-start)
                    start = time.time()

                if self.global_step.eval() % self.skip == 0:
                    # plotting
                    fig = plt.figure(1,figsize=(15,5))
                    fig.add_subplot(1,2,1)
                    plt.plot(log['net_costs'])
                    plt.title('Net cost')
                    plt.xlabel('Iterations/100')
                    fig.add_subplot(1,2,2)
                    plt.plot(log['disc_costs'])
                    plt.title('Discriminator cost')
                    plt.xlabel('Iterations/100')
                    path = os.path.join(self.working_dir,'figure')
                    plt.savefig(os.path.join(path,'train-'+str(self.global_step.eval()+1)+'.png',bbox_inches='tight',dpi=800))
                    plt.close()
                    # save model & log
                    self.save_log(log)
                    self.save_model(sess) 
                

def run(args):
    DATASET = args.db
    flags = SETUP(DATASET)
    if DATASET == 'grid':
        data_dist = Grid()
    elif DATASET == 'low_dim_embed':
        data_dist = LowDimEmbed()
    elif DATASET == 'color_mnist':
        data_dist = CMNIST(os.path.join('data','mnist'))

    noise_dist = NormalNoise(flags.NOISE_DIM)
    single_noise_dist = NormalNoise(PERTURB)
    model = AVB_AC(data_dist, noise_dist, single_noise_dist, flags, args)
    model.create_model()
    model.train()
