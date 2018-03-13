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

class AAE:
    def __init__(self, data_dist, noise_dist, flags, args):
        self.epochs = flags.EPOCHS
        self.num_samples = flags.NUM_SAMPLES
        self.learning_rate = flags.LEARNING_RATE
        self.data_dim = flags.DATA_DIM
        self.noise_dim = flags.NOISE_DIM
        self.gen_arch = flags.GEN_ARCH
        self.enc_arch = flags.ENC_ARCH
        self.batch_size = flags.BATCH_SIZE
        self.data_dist = data_dist
        self.noise_dist = noise_dist
        self.db = flags.DATASET
        self.skip = flags.SKIP
        flags.DISC_ARCH[0] = flags.NOISE_DIM
        self.disc_arch = flags.DISC_ARCH
        # setup some alternative definitions given by user
        self.hidden_acti = args.hidden_acti
        self.enc_out_acti = args.enc_out_acti
        self.gen_out_acti = args.gen_out_acti
        self.disc_out_acti = args.disc_out_acti
        self.working_dir = args.working_dir
        self.batch_norm = bool(args.batch_norm)

    def create_model(self):
        self.x = tf.placeholder(tf.float32, shape=(self.data_dim, None))
        self.z = tf.placeholder(tf.float32, shape=(self.noise_dim, None))
        self.is_train = tf.placeholder(tf.bool, shape=())

        # Generator
        generator = Generator(self.gen_arch, self.hidden_acti, self.gen_out_acti, 'GEN', self.batch_norm)
        self.fake_data = generator(self.z, train=self.is_train)

        # Encoder
        encoder = Encoder(self.enc_arch, self.hidden_acti, self.enc_out_acti, 'ENC', self.batch_norm)
        self.input = self.x
        self.true_input = encoder(self.input, train=self.is_train)

        # Discriminator
        discriminator = Discriminator(self.disc_arch, self.hidden_acti, self.disc_out_acti, 'DISC', self.batch_norm)
        self.fake_output = discriminator(self.true_input, train=self.is_train)
        self.true_output = discriminator(self.z, reuse=True, train=self.is_train)

        # Autoencoder cost (reconstruction phase)
        diff = self.x - generator(self.true_input, reuse=True)
        temp = diff**2
        dist = tf.reduce_sum(temp, axis=0, keep_dims=True)
        self.reconstruction_cost = tf.reduce_mean(dist)

        # Discriminator cost (Regularization phase)
        temp1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.true_output, labels=tf.ones_like(self.true_output))
        temp2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_output, labels=tf.zeros_like(self.fake_output))
        self.disc_cost = tf.reduce_mean(temp1) + tf.reduce_mean(temp2)

        # Encoder cost (Regularization phase)
        temp = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_output, labels=tf.ones_like(self.fake_output))
        self.enc_cost = tf.reduce_mean(temp)

        self.disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DISC')
        self.enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'ENC')
        self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'GEN')

        self.global_step = tf.Variable(0, trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.rec_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.reconstruction_cost,
                var_list=self.gen_params + self.enc_params,		
                global_step=self.global_step
            )
            self.disc_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.disc_cost,
                var_list=self.disc_params
            )
            self.enc_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.enc_cost,
                var_list=self.enc_params
            )
            
    def decode(self, z):
        return self.fake_data.eval(feed_dict={
            self.z:z,
            self.is_train:True
        })
        
    def encode(self, x):
        return self.true_input.eval(feed_dict={
            self.x:x,
            self.is_train:True
        })

    def load_model(self, sess, ckpt_id=None):
        saver = tf.train.Saver()
        path = os.path.join(self.working_dir, 'model')
        if ckpt_id:
            ckpt =  path + '/saved-model-' + str(ckpt_id)
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
        path = os.path.join(self.working_dir, 'model/saved-model')
        save_path = saver.save(sess, path, global_step=self.global_step.eval()+1)
        print('\nModel saved in %s'%save_path)
    
    def save_log(self,log):
        path = os.path.join(self.working_dir,'model/log.pkl')
        with open(path,'wb') as f:
            pickle.dump(log,f)

    def load_log(self,log):
        path = os.path.join(self.working_dir,'model/log.pkl')
        if os.path.exists(path):
            with open(path,'rb') as f:
                data = pickle.load(f)
                log['enc_costs'] = data['enc_costs']
                log['disc_costs'] = data['disc_costs']
                log['rec_costs'] = data['rec_costs']
                log['train_time'] = data['train_time']

    def train(self):
        true_data,_ = self.data_dist.sample(self.batch_size)

        init = tf.global_variables_initializer()
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True

        log = {'enc_costs':[],'disc_costs':[],'rec_costs':[],'train_time':0}
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
                if self.db not in ['grid','ring','low_dim_embed']:
                    true_data,_ = self.data_dist.sample(self.batch_size)
                noise = self.noise_dist.sample(self.batch_size)
                #eta = single_noise_dist.sample(self.batch_size)
                sess.run(self.rec_opt, feed_dict={
                    self.z: noise,
                    self.x: true_data,
                    self.is_train: True
                })
                sess.run(self.disc_opt, feed_dict={
                    self.z: noise,
                    self.x: true_data,
                    self.is_train: True
                })  
                sess.run(self.enc_opt, feed_dict={
                    self.z: noise,
                    self.x: true_data,
                    self.is_train: True
                })          
                enc_cost, disc_cost, rec_cost = sess.run([self.enc_cost,self.disc_cost,self.reconstruction_cost], feed_dict={
                    self.z: noise,
                    self.x: true_data,
                    self.is_train: True
                })
                
                if self.global_step.eval() % 100 == 0: # saving at every iteration is not neccessary
                    log['enc_costs'].append(enc_cost)
                    log['disc_costs'].append(disc_cost)
                    log['rec_costs'].append(rec_cost)
                    log['train_time'] += (time.time()-start)
                    start = time.time()

                if self.global_step.eval() % self.skip == 0:
                    # plotting
                    fig = plt.figure(1,figsize=(15,10))
                    fig.add_subplot(2,2,1)
                    plt.plot(log['rec_costs'])
                    plt.title('Reconstruction cost')
                    plt.xlabel('Iterations/100')
                    fig.add_subplot(2,2,2)
                    plt.plot(log['disc_costs'])
                    plt.title('Discriminator cost')
                    plt.xlabel('Iterations/100')
                    fig.add_subplot(2,2,3)
                    plt.plot(log['enc_costs'])
                    plt.title('Encoder cost')
                    plt.xlabel('Iterations/100')
                    path = os.path.join(self.working_dir,'figure')
                    plt.savefig(path+'/train-'+str(self.global_step.eval()+1)+'.png',bbox_inches='tight',dpi=800)
                    plt.close()
                    # save model & log
                    self.save_log(log)
                    self.save_model(sess)                   
                    

def run(args):
    DATASET = args.db
    flags = SETUP(DATASET)
    flags.LEARNING_RATE = 1e-3
    if DATASET == 'grid':
        data_dist = Grid()
    elif DATASET == 'low_dim_embed':
        data_dist = LowDimEmbed()
    elif DATASET == 'color_mnist':
        flags.LEARNING_RATE = 1e-4
        data_dist = CMNIST('data/mnist')

    noise_dist = UniformNoise(flags.NOISE_DIM)
    model = AAE(data_dist, noise_dist, flags, args)
    model.create_model()
    model.train()
