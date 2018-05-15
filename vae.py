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
import time
import pickle
from tqdm import tqdm

class VAE:
    def __init__(self, data_dist, noise_dist, flags, args):
        self.epochs = flags.EPOCHS
        self.num_samples = flags.NUM_SAMPLES
        self.batch_size = flags.BATCH_SIZE
        self.learning_rate = flags.LEARNING_RATE
        self.data_dim = flags.DATA_DIM
        self.noise_dim = flags.NOISE_DIM
        self.gen_arch = flags.GEN_ARCH
        flags.ENC_ARCH[-1] = flags.ENC_ARCH[-1]*2
        self.enc_arch = flags.ENC_ARCH
        self.data_dist = data_dist
        self.noise_dist = noise_dist
        self.db = flags.DATASET
        self.skip = flags.SKIP
        # setup some alternative definitions given by user
        self.hidden_acti = args.hidden_acti
        self.enc_out_acti = args.enc_out_acti
        self.gen_out_acti = args.gen_out_acti
        self.disc_out_acti = args.disc_out_acti
        self.working_dir = args.working_dir
        self.GPU = args.gpu

    def create_model(self):
        with tf.device('/device:GPU:'+str(self.GPU)):
            self.x = tf.placeholder(tf.float32, shape=(self.data_dim, None))
            self.z = tf.placeholder(tf.float32, shape=(self.noise_dim, None))
            self.epsilon = tf.placeholder(tf.float32, shape=(self.noise_dim, None))

            # Generator
            generator = Generator(self.gen_arch, self.hidden_acti, self.gen_out_acti, 'GEN')
            self.fake_data = generator(self.z)

            # Encoder
            encoder = Encoder(self.enc_arch, self.hidden_acti, self.enc_out_acti, 'ENC')
            self.true_input = encoder(self.x)
            self.mean, self.std = tf.split(self.true_input, num_or_size_splits=2, axis=0)

            # Reconstruction
            self.z_sampled = self.mean + self.std*self.epsilon
            self.true_output = generator(self.z_sampled, reuse=True)
            diff = self.x - self.true_output
            temp = diff**2
            dist = tf.reduce_sum(temp, axis=0, keep_dims=True)
            self.reconstruction = tf.reduce_mean(dist)
                
            # Kullback Leibler term
            kullback = self.mean**2 + self.std**2 - 1.0 - tf.log(self.std**2)
            self.kullback = tf.reduce_mean(tf.reduce_sum(kullback, axis=0, keep_dims=True))

            # Total cost
            self.cost = 0.5*self.reconstruction + 0.5*self.kullback

            self.all_params = tf.trainable_variables()
            self.global_step = tf.Variable(0, trainable=False)
            self.cost_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.cost,
                var_list=self.all_params,
                global_step=self.global_step
            )
    def decode(self, z):
        return self.fake_data.eval(feed_dict={
            self.z:z
        })
        
    def encode(self, x):
        return self.true_input.eval(feed_dict={
            self.x:x
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
                log['kullback_costs'] = data['kullback_costs']
                log['rec_costs'] = data['rec_costs']
                log['train_time'] = data['train_time']

    def train(self):
        true_data,_ = self.data_dist.sample(self.batch_size)

        init = tf.global_variables_initializer()
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True

        log = {'kullback_costs':[],'rec_costs':[],'train_time':0}
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
                epsilon = self.noise_dist.sample(self.batch_size)
                kullback_cost, _ = sess.run([self.kullback, self.cost_opt], feed_dict={
                    self.z: noise,
                    self.x: true_data,
                    self.epsilon: epsilon
                })
                rec_cost = sess.run(self.reconstruction, feed_dict={
                    self.x: true_data,
                    self.z: noise,
                    self.epsilon: epsilon
                })
                
                if self.global_step.eval() % 100 == 0: # saving at every iteration is not neccessary
                    log['kullback_costs'].append(kullback_cost)
                    log['rec_costs'].append(rec_cost)
                    log['train_time'] += (time.time()-start)
                    start = time.time()

                if self.global_step.eval() % self.skip == 0:
                    # plotting
                    fig = plt.figure(1,figsize=(15,5))
                    fig.add_subplot(2,2,1)
                    plt.plot(log['rec_costs'])
                    plt.title('Reconstruction cost')
                    plt.xlabel('Iterations/100')
                    fig.add_subplot(2,2,2)
                    plt.plot(log['kullback_costs'])
                    plt.title('Kullback Leibler cost')
                    plt.xlabel('Iterations/100')
                    path = os.path.join(self.working_dir,'figure')
                    plt.savefig(os.path.join(path,'train-'+str(self.global_step.eval()+1)+'.png'),bbox_inches='tight',dpi=800)
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
        data_dist = CMNIST(os.path.join('data','mnist'))
    elif DATASET == 'cifar_100':
        flags.LEARNING_RATE = 1e-4
        data_dist = CIFAR100(os.path.join('data', 'cifar-100'))

    noise_dist = NormalNoise(flags.NOISE_DIM)
    model = VAE(data_dist, noise_dist, flags, args)
    model.create_model()
    model.train()
