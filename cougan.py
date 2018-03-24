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
from generator import Generator
from discriminator import Discriminator
from encoder import Encoder
import time
import pickle
from tqdm import tqdm

class CoulombGAN:
    def __init__(self, data_dist, noise_dist, flags, args):
        self.epochs = flags.EPOCHS
        self.num_samples = flags.NUM_SAMPLES
        self.disc_learning_rate = 5e-5
        self.gen_learning_rate = 1e-4
        self.enc_learning_rate = 1e-4
        self.exponential_decay = flags.EXPONENTIAL_DECAY
        self.data_dim = flags.DATA_DIM
        self.noise_dim = flags.NOISE_DIM
        self.gen_arch = flags.GEN_ARCH
        self.enc_arch = flags.ENC_ARCH
        flags.DISC_ARCH[0] = flags.DATA_DIM
        self.disc_arch = flags.DISC_ARCH
        self.batch_size = flags.BATCH_SIZE
        self.kernel_dimension = 3.0
        self.epsilon = 1.0
        self.LAMBDA = 1e-7
        self.data_dist = data_dist
        self.noise_dist = noise_dist
        self.db = flags.DATASET
        self.skip = flags.SKIP
        # setup some alternative definitions given by user
        self.hidden_acti = args.hidden_acti
        self.disc_out_acti = args.disc_out_acti
        self.gen_out_acti = args.gen_out_acti
        self.enc_out_acti = args.enc_out_acti
        self.working_dir = args.working_dir
        self.train_enc_steps = args.train_enc_steps
        self.batch_norm = bool(args.batch_norm)
        if args.db == 'grid':
            self.disc_learning_rate = 1e-3
            self.gen_learning_rate = 1e-3
            self.enc_learning_rate = 1e-3
            self.start_epsilon = 3.0
        elif args.db == 'low_dim_embed':
            self.disc_learning_rate = 1e-4
            self.gen_learning_rate = 1e-4
            self.enc_learning_rate = 1e-4
            self.start_epsilon = 3.0

    def plummer(self, a, b):
        temp1 = tf.transpose(tf.reduce_sum(a**2, axis=0, keep_dims=True))
        temp2 = tf.reduce_sum(b**2, axis=0, keep_dims=True)
        temp3 = tf.matmul(tf.transpose(a), b)
        dist = temp1 + temp2 - 2.0*temp3
        return 1.0/tf.sqrt(dist + self.epsilon**2)**(self.kernel_dimension-2)

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
        self.fake_input = encoder(self.fake_data, reuse=True, train=self.is_train)

        # Discriminator
        discriminator = Discriminator(self.disc_arch, self.hidden_acti, self.disc_out_acti, 'DISC', self.batch_norm)
        self.true_output = discriminator(self.x, train=self.is_train)
        self.fake_output = discriminator(self.fake_data, reuse=True, train=self.is_train)

        # Similarity matrices
        true_true = self.plummer(self.x, self.x)
        true_fake = self.plummer(self.x, self.fake_data)
        fake_true = tf.transpose(true_fake)
        fake_fake = self.plummer(self.fake_data, self.fake_data)    

        regularizers = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # Cost functions
        dist1 = tf.transpose(self.fake_output) \
                - tf.reduce_mean(fake_true, axis=1, keep_dims=True) \
                + tf.reduce_mean(fake_fake, axis=1, keep_dims=True)
        dist2 = tf.transpose(self.true_output) \
                - tf.reduce_mean(true_true, axis=1, keep_dims=True) \
                + tf.reduce_mean(true_fake, axis=1, keep_dims=True)
        self.disc_cost = 0.5*tf.reduce_mean(dist1**2) + 0.5*tf.reduce_mean(dist2**2) + self.LAMBDA*regularizers
        self.gen_cost = - 0.5*tf.reduce_mean(self.fake_output) + self.LAMBDA*regularizers
        dist = tf.reduce_sum(tf.square(self.z - self.fake_input), axis=0)
        self.enc_cost = tf.reduce_mean(dist)

        self.disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DISC')
        self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'GEN')
        self.enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'ENC')

        self.disc_global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.exponential_decay == True:
                n_iters = int(self.num_samples/self.batch_size)*self.epochs
                start_decayed = int(n_iters/2)
                if self.db in ['grid', 'low_dim_embed']:
                    start_decayed = 1000
                disc_decayed_lr = tf.where(
                    #decayed_learning_rate = learning_rate * decay_rate^(global_step/decay_steps)
                    tf.greater_equal(self.disc_global_step, start_decayed),
                    tf.train.exponential_decay(self.disc_learning_rate,\
                                            global_step=self.disc_global_step,\
                                            decay_steps=10000,\
                                            decay_rate=0.9),
                    self.disc_learning_rate
                )
                disc_learning_rate = tf.where(tf.less(disc_decayed_lr, 2e-6), 2e-6, disc_decayed_lr)
                self.disc_opt = tf.train.AdamOptimizer(disc_learning_rate).minimize(
                    self.disc_cost,
                    var_list=self.disc_params,
                    global_step=self.disc_global_step
                )
                gen_decayed_lr = tf.where(
                            #decayed_learning_rate = learning_rate * decay_rate^(global_step/decay_steps)
                            tf.greater_equal(self.global_step, start_decayed),
                            tf.train.exponential_decay( self.gen_learning_rate,\
                                                        global_step=self.global_step,\
                                                        decay_steps=10000,\
                                                        decay_rate=0.9),
                            self.gen_learning_rate
                )
                gen_learning_rate = tf.where(tf.less(gen_decayed_lr, 2e-6), 2e-6, gen_decayed_lr)
                self.gen_opt = tf.train.AdamOptimizer(gen_learning_rate).minimize(
                    self.gen_cost,
                    var_list=self.gen_params,
                    global_step=self.global_step
                )
                # self.disc_opt = tf.train.AdamOptimizer(self.disc_learning_rate).minimize(
                #     self.disc_cost,
                #     var_list=self.disc_params,
                #     global_step=self.disc_global_step
                # )
                # self.gen_opt = tf.train.AdamOptimizer(self.gen_learning_rate).minimize(
                #     self.gen_cost,
                #     var_list=self.gen_params,
                #     global_step=self.global_step
                # )
                # self.enc_opt = tf.train.AdamOptimizer(self.enc_learning_rate).minimize(
                #     self.enc_cost,
                #     var_list=self.enc_params,
                #     global_step=self.global_step
                # )
                if self.db in ['grid', 'low_dim_embed']:
                    decayed_epsilon = tf.where(
                            #decayed_learning_rate = learning_rate * decay_rate^(global_step/decay_steps)
                            tf.greater_equal(self.global_step, start_decayed),
                            tf.train.exponential_decay(self.start_epsilon,\
                                                    global_step=self.global_step,\
                                                    decay_steps=10000,\
                                                    decay_rate=0.9),
                            self.start_epsilon
                    )
                    self.epsilon = tf.where(tf.less(decayed_epsilon, 1.0), 1.0, decayed_epsilon)
            else:
                self.disc_opt = tf.train.AdamOptimizer(self.disc_learning_rate).minimize(
                    self.disc_cost,
                    var_list=self.disc_params,
                    global_step=self.disc_global_step
                )
                self.gen_opt = tf.train.AdamOptimizer(self.gen_learning_rate).minimize(
                    self.gen_cost,
                    var_list=self.gen_params,
                    global_step=self.global_step
                )
                self.enc_opt = tf.train.AdamOptimizer(self.enc_learning_rate).minimize(
                    self.enc_cost,
                    var_list=self.enc_params,
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
            ckpt =  os.path.join(path,'saved-model-' + str(ckpt_id))
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
                log['gen_costs'] = data['gen_costs']
                log['enc_costs'] = data['enc_costs']
                log['train_time'] = data['train_time']

    def train(self):
        true_data,_ = self.data_dist.sample(self.batch_size)

        init = tf.global_variables_initializer()
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        log = {'disc_costs':[],'gen_costs':[],'enc_costs':[],'train_time':0}
        start = time.time()
        with tf.Session(config=config) as sess:
            sess.run(init)
            # model loading
            self.load_model(sess)
            self.load_log(log)

            # how many iters?
            n_iters = int(self.num_samples/self.batch_size)*self.epochs
            if self.train_enc_steps == 0:
                n_iters_left = n_iters - self.global_step.eval()
            else:
                n_iters_left = self.train_enc_steps

            print('\nNumber of train iterations %d'%n_iters_left)

            for t in tqdm(range(n_iters_left)):
                if self.db not in ['grid','ring','low_dim_embed']:
                    true_data,_ = self.data_dist.sample(self.batch_size)
                noise = self.noise_dist.sample(self.batch_size)
                if self.train_enc_steps == 0:
                    _,_ = sess.run([self.disc_opt, self.gen_opt], feed_dict={
                        self.x: true_data,
                        self.z: noise,
                        self.is_train: True
                    })
                else:
                    _ = sess.run(self.enc_opt, feed_dict={
                        self.x: true_data,
                        self.z: noise,
                        self.is_train: True
                    })
                disc_cost, gen_cost, enc_cost = sess.run([self.disc_cost, self.gen_cost, self.enc_cost], feed_dict={
                    self.x: true_data,
                    self.z: noise,
                    self.is_train: True
                })
                
                check_nan = np.any(np.isnan(disc_cost))
                assert not check_nan, 'disc cost is nan'
                check_nan = np.any(np.isnan(gen_cost))
                assert not check_nan, 'gen cost is nan'
                check_nan = np.any(np.isnan(enc_cost))
                assert not check_nan, 'enc cost is nan'

                if self.global_step.eval() % 100 == 0: # saving at every iteration is not neccessary
                    log['disc_costs'].append(disc_cost)
                    log['gen_costs'].append(gen_cost)
                    log['enc_costs'].append(enc_cost)
                    log['train_time'] += (time.time()-start)
                    start = time.time()
                if self.global_step.eval() % self.skip == 0:
                    # plotting
                    fig = plt.figure(1,figsize=(15,10))
                    fig.add_subplot(2,2,1)
                    plt.plot(log['disc_costs'])
                    plt.title('Discriminator cost')
                    plt.xlabel('Iterations/100')
                    fig.add_subplot(2,2,2)
                    plt.plot(log['gen_costs'])
                    plt.title('Generator cost')
                    plt.xlabel('Iterations/100')
                    fig.add_subplot(2,2,3)
                    plt.plot(log['enc_costs'])
                    plt.title('Encoder cost')
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
        flags.EXPONENTIAL_DECAY = True
        data_dist = Grid()
    elif DATASET == 'low_dim_embed':
        flags.EXPONENTIAL_DECAY = True
        data_dist = LowDimEmbed()
    elif DATASET == 'color_mnist':
        data_dist = CMNIST(os.path.join('data','mnist'))
    
    noise_dist = NormalNoise(flags.NOISE_DIM)
    model = CoulombGAN(data_dist, noise_dist, flags, args)
    model.create_model()
    model.train()
