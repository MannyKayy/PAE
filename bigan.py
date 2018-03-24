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

class BiGAN:
    def __init__(self, data_dist, noise_dist, flags, args):
        self.epochs = flags.EPOCHS
        self.num_samples = flags.NUM_SAMPLES
        self.batch_size = flags.BATCH_SIZE
        self.learning_rate = flags.LEARNING_RATE
        self.exponential_decay = flags.EXPONENTIAL_DECAY
        self.data_dim = flags.DATA_DIM
        self.noise_dim = flags.NOISE_DIM
        self.gen_arch = flags.GEN_ARCH
        self.enc_arch = flags.ENC_ARCH
        self.disc_arch = flags.DISC_ARCH
        self.data_dist = data_dist
        self.noise_dist = noise_dist
        self.db = flags.DATASET
        self.skip = flags.SKIP
        if self.db in ['grid','low_dim_embed']:
            self.LAMBDA = 0
        else:
            self.LAMBDA = 2.5*1e-5
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
        self.true_input = encoder(self.x, train=self.is_train)

        # Discriminator
        self.true_pair = tf.concat([self.x,self.true_input], 0)
        self.fake_pair = tf.concat([self.fake_data,self.z], 0)
        discriminator = Discriminator(self.disc_arch, self.hidden_acti, self.disc_out_acti, 'DISC', self.batch_norm)
        self.true_output = discriminator(self.true_pair, train=self.is_train)
        self.fake_output = discriminator(self.fake_pair, reuse=True, train=self.is_train)

        # Total cost
        temp1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_output, labels=tf.ones_like(self.fake_output))
        temp2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.true_output, labels=tf.zeros_like(self.true_output))
        regularizers = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.cost = tf.reduce_mean(temp1) + tf.reduce_mean(temp2)

        self.disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DISC')
        self.enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'ENC')
        self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'GEN')

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.exponential_decay == True:
                n_iters = int(self.num_samples/self.batch_size)*self.epochs
                decayed_lr = tf.where(
                            #decayed_learning_rate = learning_rate * decay_rate^(global_step/decay_steps)
                            tf.greater_equal(self.global_step, int(n_iters/2)),
                            tf.train.exponential_decay( self.learning_rate,\
                                                        global_step=self.global_step,\
                                                        decay_steps=10000,\
                                                        decay_rate=0.9),
                            self.learning_rate)
                learning_rate = tf.where(tf.less(decayed_lr, 2e-6), 2e-6, decayed_lr)
                self.disc_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999).minimize(
                    self.cost + self.LAMBDA*regularizers,
                    var_list=self.disc_params
                )
                self.net_opt = tf.train.AdamOptimizer(learning_rate).minimize(
                    -self.cost + self.LAMBDA*regularizers,
                    var_list=self.gen_params+self.enc_params,
                    global_step=self.global_step
                )
            else:
                self.disc_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    self.cost + self.LAMBDA*regularizers,
                    var_list=self.disc_params
                )
                self.net_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    -self.cost + self.LAMBDA*regularizers,
                    var_list=self.gen_params+self.enc_params,
                    global_step=self.global_step
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
                log['disc_costs'] = data['disc_costs']
                log['net_costs'] = data['net_costs']
                log['train_time'] = data['train_time']

    def train(self):
        true_data,_ = self.data_dist.sample(self.batch_size)

        init = tf.global_variables_initializer()
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        log = {'disc_costs':[],'net_costs':[],'train_time':0}
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
                _, _ = sess.run([self.disc_opt, self.net_opt], feed_dict={
                    self.z: noise,
                    self.x: true_data,
                    self.is_train: True
                })
                net_temp = sess.run(self.cost, feed_dict={
                    self.x: true_data,
                    self.z: noise,
                    self.is_train: True
                })
                
                if self.global_step.eval() % 100 == 0: # saving at every iteration is not neccessary
                    log['disc_costs'].append(net_temp)
                    log['net_costs'].append(-net_temp)
                    log['train_time'] += (time.time()-start)
                    start = time.time()
                
                if self.global_step.eval() % self.skip == 0:
                    # plotting
                    fig = plt.figure(1,figsize=(15,5))
                    fig.add_subplot(1,2,1)
                    plt.plot(log['disc_costs'])
                    plt.title('Discriminator cost')
                    plt.xlabel('Iterations/100')
                    fig.add_subplot(1,2,2)
                    plt.plot(log['net_costs'])
                    plt.title('Net cost')
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
    flags.LEARNING_RATE = 2e-4
    flags.EXPONENTIAL_DECAY = True
    if DATASET == 'grid':
        data_dist = Grid()
    elif DATASET == 'low_dim_embed':
        data_dist = LowDimEmbed()
    elif DATASET == 'color_mnist':
        data_dist = CMNIST(os.path.join('data','mnist'))

    noise_dist = UniformNoise(flags.NOISE_DIM)
    model = BiGAN(data_dist, noise_dist, flags, args)
    model.create_model()
    model.train()
