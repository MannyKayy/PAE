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
slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace
import time
import pickle
from tqdm import tqdm

class VEEGAN:
    def __init__(self, data_dist, noise_dist, input_noise_dist, flags, args):
        self.epochs = flags.EPOCHS
        self.num_samples = flags.NUM_SAMPLES
        self.batch_size = flags.BATCH_SIZE
        self.learning_rate = flags.LEARNING_RATE
        self.exponential_decay = flags.EXPONENTIAL_DECAY
        self.data_dim = flags.DATA_DIM
        self.noise_dim = flags.NOISE_DIM
        flags.GEN_ARCH[0] = flags.GEN_ARCH[0] + 1
        self.gen_arch = flags.GEN_ARCH
        self.enc_arch = flags.ENC_ARCH
        self.disc_arch = flags.DISC_ARCH
        self.data_dist = data_dist
        self.noise_dist = noise_dist
        self.input_noise_dist = input_noise_dist
        self.db = flags.DATASET
        self.skip = flags.SKIP
        # setup some alternative definitions given by user
        self.hidden_acti = args.hidden_acti
        self.disc_out_acti = args.disc_out_acti
        self.gen_out_acti = args.gen_out_acti
        self.enc_out_acti = args.enc_out_acti
        self.working_dir = args.working_dir
        self.batch_norm = bool(args.batch_norm)

    def standard_normal(self, shape, **kwargs):
        return st.StochasticTensor(ds.MultivariateNormalDiag(loc=tf.zeros(shape), scale_diag=tf.ones(shape), **kwargs))
    
    def create_model(self):
        self.x = tf.placeholder(tf.float32, shape=(self.data_dim, None))
        self.z = tf.placeholder(tf.float32, shape=(self.noise_dim, None))
        self.is_train = tf.placeholder(tf.bool, shape=())
        
        self.input_noise = tf.placeholder(tf.float32, shape=(1, None))        
        self.output_noise = tf.placeholder(tf.float32, shape=(self.noise_dim, None))

        # Generator
        generator = Generator(self.gen_arch, self.hidden_acti, self.gen_out_acti, 'GEN', self.batch_norm)
        self.fake_data = generator(tf.concat([self.z, self.input_noise], 0)) # Stochastic net with noise in the input

        # Encoder
        encoder = Encoder(self.enc_arch, self.hidden_acti, self.enc_out_acti, 'ENC', self.batch_norm)
        self.true_input = encoder(self.x) + self.output_noise # Gaussian output centered at the output of the encoder and identity covariance

        # Discriminator
        self.true_pair = tf.concat([self.x,self.true_input], 0)
        self.fake_pair = tf.concat([self.fake_data,self.z], 0)
        discriminator = Discriminator(self.disc_arch, self.hidden_acti, self.disc_out_acti, 'DISC', self.batch_norm)
        self.true_output = discriminator(self.true_pair)
        self.fake_output = discriminator(self.fake_pair, reuse=True)

        # Reconstruction
        self.reconstructed_output = encoder(self.fake_data, reuse=True)
        diff = self.z - self.reconstructed_output
        temp = diff**2
        dist = tf.reduce_sum(temp, axis=0, keep_dims=True)
        self.reconstruction = tf.reduce_mean(dist)

        # Total cost
        temp1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_output, labels=tf.ones_like(self.fake_output))
        temp2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.true_output, labels=tf.zeros_like(self.true_output))
        self.disc_cost = tf.reduce_mean(temp1) + tf.reduce_mean(temp2)
        self.net_cost = tf.reduce_mean(self.fake_output) + self.reconstruction
        
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
                self.disc_opt = tf.train.AdamOptimizer(learning_rate, beta1=.5).minimize(
                    self.disc_cost,
                    var_list=self.disc_params
                )
                self.net_opt = tf.train.AdamOptimizer(learning_rate, beta1=.5).minimize(
                    self.net_cost,
                    var_list=self.gen_params+self.enc_params,
                    global_step=self.global_step
                )
            else:
                self.disc_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=.5).minimize(
                    self.disc_cost,
                    var_list=self.disc_params
                )
                self.net_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=.5).minimize(
                    self.net_cost,
                    var_list=self.gen_params+self.enc_params,
                    global_step=self.global_step
                )

    def decode(self, z):
        input_noise = self.input_noise_dist.sample(z.shape[1])
        return self.fake_data.eval(feed_dict={
            self.z:z,
            self.input_noise:input_noise,
            self.is_train:True
        })
    
    def encode(self, x):
        output_noise = self.noise_dist.sample(x.shape[1])
        return self.true_input.eval(feed_dict={
            self.x:x,
            self.is_train:True,
            self.output_noise:output_noise
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
                input_noise = self.input_noise_dist.sample(self.batch_size)
                output_noise = self.noise_dist.sample(self.batch_size)
                _, _ = sess.run([self.disc_opt, self.net_opt], feed_dict={
                    self.z: noise,
                    self.x: true_data,
                    self.input_noise: input_noise,
                    self.output_noise: output_noise,
                    self.is_train: True            
                }) 
                
                disc_temp, net_temp = sess.run([self.disc_cost,self.net_cost], feed_dict={
                    self.z: noise,
                    self.x: true_data,
                    self.input_noise: input_noise,
                    self.output_noise: output_noise,
                    self.is_train: True             
                })

                if self.global_step.eval() % 100 == 0: # saving at every iteration is not neccessary
                    log['disc_costs'].append(disc_temp)
                    log['net_costs'].append(net_temp)
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
                    plt.savefig(path+'/train-'+str(self.global_step.eval()+1)+'.png',bbox_inches='tight',dpi=800)
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
        data_dist = CMNIST('data/mnist')
    
    if DATASET == 'grid':
        flags.NOISE_DIM = 254 # refer to author's implementation
        flags.GEN_ARCH[0] = flags.NOISE_DIM
        flags.ENC_ARCH[-1] = flags.NOISE_DIM
        flags.DISC_ARCH[0] = flags.NOISE_DIM + flags.DATA_DIM
        #flags.DISC_ARCH = [flags.NOISE_DIM+flags.DATA_DIM, 128, 1]
        flags.LEARNING_RATE = 1e-3
    elif DATASET == 'low_dim_embed':
        flags.DISC_ARCH = [flags.NOISE_DIM+flags.DATA_DIM, 128, 128, 1]
        flags.LEARNING_RATE = 1e-3
    else:
        flags.LEARNING_RATE = 2e-4 # follow DCGAN
        flags.EXPONENTIAL_DECAY = True
    noise_dist = UniformNoise(flags.NOISE_DIM)
    input_noise_dist = UniformNoise(1)
    model = VEEGAN(data_dist, noise_dist, input_noise_dist, flags, args)
    model.create_model()
    model.train()
    