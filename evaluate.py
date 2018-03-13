from __future__ import division
import argparse
import os
from data import *
import tensorflow as tf
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import mixture
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from classifier import *
import fid
import itertools
import platform
import matplotlib
import matplotlib.pyplot as plt
if 'Linux' in platform.platform():
    plt.switch_backend('agg')


class Evaluator:
    def __init__(self, model):
        self.model = model
    
    def visualize(self, gen_data, ckpt_id):
        '''
            This function visualizes data
            Each distribution has it own data visualization
        '''
        working_dir = self.model.working_dir
        data_dist.visualize(gen_data, save_path=os.path.join(working_dir,'figure/') + DATASET + '-' + str(ckpt_id) + '.png')
    
    def visualize_closest(self, test_data, gen_data, ckpt_id):
        '''
            This function visualizes generated data closest to real data
            Each distribution has it own data visualization
        '''
        working_dir = self.model.working_dir
        data_dist.visualize(gen_data, test_data, save_path=os.path.join(working_dir,'figure/') + DATASET + '-nr-' + str(ckpt_id) + '.png')

    def compute_log_likelihood(self, val_gen_data, gen_data, ckpt_id, batch_size=10000):
        # perform cross validation on val data to select kernel bandwidth
        params = {'bandwidth': np.logspace(-3.0, 1.5, 10, base=10.0)}
        grid = GridSearchCV(KernelDensity(), params, n_jobs=-1)
        grid.fit(val_gen_data.T) # NUM_SAMPLES x NUM_FEATURES
        best_sigma_fake = grid.best_estimator_.bandwidth
        print('Best sigma %.3f' % best_sigma_fake)
        # log-likelihood of test data under gen distribution
        kde_fake = KernelDensity(bandwidth=best_sigma_fake)
        kde_fake.fit(gen_data.T)
        n = gen_data.shape[1]
        n_batches = n//batch_size
        llh = np.zeros(n_batches)
        test_data,_ = data_dist.test_sample(n)
        for i in range(n_batches):
            start = i*batch_size
            end = start + batch_size
            scores = kde_fake.score_samples(test_data[:,start:end].T)
            llh[i] = np.mean(scores)
        print('log-likelihood of real data under model distribution: %.3f +- %.3f'\
            %(np.mean(llh), np.std(llh)))

    def is_pos_def(self,x):
        return np.all(np.linalg.eigvals(x) > 0)

    def reshape_data(self, x, db):
        '''
            This function reshapes data so that it is compatible for CNN
        '''
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

    def classify_cmnist(self, model_path, x):
        saver = tf.train.Saver()
        path = os.path.join(model_path)
        ckpt = tf.train.latest_checkpoint(path)
        if ckpt:
            g = tf.Graph()
            with g.as_default():
                clf = CNN_CLF(None, 28, 28, 3, 1000)
                clf.create_model()
                sess = tf.Session(graph=g)
                saver = tf.train.Saver()
                saver.restore(sess, ckpt)
                # reshape data before passing to CNN
                x_new = np.zeros((x.shape[1],28,28,3))
                for i in range(x.shape[1]):
                    for c in range(3):
                        x_new[i,:,:,c] = x[c*28*28:(c+1)*28*28,i].reshape(28,28)
                predictions = clf.predict(x_new, sess)
                sess.close()
                return predictions
        else:
            print('\nModel not found')
            return None

    def classify_low_dim_embed(self, model_path, x):
        path = os.path.join(model_path)
        ckpt = tf.train.latest_checkpoint(path)
        if ckpt:
            g = tf.Graph()
            with g.as_default():
                clf = MLP_CLF(None, ARCH_MLP)
                clf.create_model()
                sess = tf.Session(graph=g)
                saver = tf.train.Saver()
                saver.restore(sess, ckpt)
                predictions = clf.predict(x, sess)
                sess.close()
                return predictions
        else:
            print('\nModel not found')
            return None
    
    def compute_fid(self, images, inception_path, batch_size=100):
        g = tf.Graph()
        with g.as_default():
            fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
            sess = tf.Session(graph=g)
            mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=batch_size)
            sess.close()
        return mu, sigma
    
    def reshape_cmnist(self, data):
        # reshape data to [batch_size, 28, 28, 3]
        data_new = np.zeros((data.shape[1],28,28,3))
        for i in range(data.shape[1]):
            for c in range(3):
                data_new[i,:,:,c] = data[c*28*28:(c+1)*28*28,i].reshape(28,28)
        return data_new

    def run(self, model_path, ckpt_ids):
        '''
            This is the main function for evaluation
            Input:
                model_path: the folder containing all checkpoints
                ckpt_ids: the list of integers indicating checkpoint ids
                that we want to evaluate
        '''
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        model = self.model
        with tf.Session(config=config) as sess:
            # start evaluating every checkpoint
            for id in ckpt_ids:     
                ckpt = os.path.join(model_path, 'saved-model-'+str(id))
                saver = tf.train.Saver()
                saver.restore(sess, ckpt)
                
                if flags.DATASET == 'grid':
                    #---------------------------------------------------------------#
                    # visualization real and random generated samples               #
                    #---------------------------------------------------------------#
                    print('\nVisualizing generated samples at iter %d'%(id))
                    gen_data = model.decode(noise_dist.sample(1000))
                    self.visualize(gen_data,id)

                if flags.DATASET == 'color_mnist':
                    #---------------------------------------------------------------#
                    # visualization real and random generated samples               #
                    #---------------------------------------------------------------#
                    print('\nVisualizing generated samples at iter %d'%(id))
                    gen_data = model.decode(noise_dist.sample(model.batch_size))
                    self.visualize(gen_data,id)
                
                if DATASET in ['color_mnist'] and args.method not in ['cougan','wgan']:
                    #---------------------------------------------------------------#
                    # visualization real and nearest generated samples              #
                    #---------------------------------------------------------------#
                    print('\nVisualizing nearest neighbors at iter %d'%(model.global_step.eval()+1))
                    real_data,_ = data_dist.sample(10)
                    
                    true_input = model.encode(real_data)
                    if args.method == 'vae':
                        [mean,std] = np.split(true_input, 2, axis=0)
                        epsilon = noise_dist.sample(10)
                        z_sampled = mean + std*epsilon
                    else:
                        z_sampled = true_input

                    gen_data = np.zeros((real_data.shape[0],90))
                    for i in range(10):
                        perturbed_input = np.random.multivariate_normal(z_sampled[:,i],0.01*np.eye(z_sampled.shape[0]),9)
                        gen_data[:,i*9:i*9+9] = model.decode(perturbed_input.T)
                    self.visualize_closest(real_data,gen_data,id)
                
                log = {}
                if DATASET == 'low_dim_embed':
                    #---------------------------------------------------------------#
                    # counting the number of modes using pretrained MLP net          #
                    #---------------------------------------------------------------#
                    print('\nClassifying generated data at iter %d'%(id))
                    gen_data = model.decode(noise_dist.sample(10000))
                    predictions = self.classify_low_dim_embed('classification/low_dim_embed/model', gen_data)
                    hist,_ = np.histogram(predictions)
                    log['hist'] = hist
                    if not os.path.exists(os.path.join(args.working_dir,'others')):
                        os.makedirs(os.path.join(args.working_dir,'others'))
                    path = os.path.join(args.working_dir,'others/hist_modes.pkl')
                    with open(path,'wb') as f:
                        pickle.dump(log,f)

                if DATASET  == 'color_mnist':
                    #---------------------------------------------------------------#
                    # counting the number of modes using pretrained CNN          #
                    #---------------------------------------------------------------#
                    print('\nClassifying generated data at iter %d'%(id))
                    hist = 0
                    for i in range(100):
                        gen_data = model.decode(noise_dist.sample(10000))
                        predictions = self.classify_cmnist('classification/color_mnist/model', gen_data)
                        h,_ = np.histogram(predictions, bins=1000)
                        hist = hist + h
                    print('number of class having 0 samples %d'%np.sum(hist==0))
                    print('mean #samples per class %d'%np.mean(hist))
                    print('std #samples per class %d'%np.std(hist))
                    log['hist'] = hist
                    if not os.path.exists(os.path.join(args.working_dir,'others')):
                        os.makedirs(os.path.join(args.working_dir,'others'))
                    path = os.path.join(args.working_dir,'others/hist_modes.pkl')
                    with open(path,'wb') as f:
                        pickle.dump(log,f)
                
                if flags.DATASET in ['grid','low_dim_embed']:
                    #---------------------------------------------------------------#
                    # compute log-likelihood                                        #
                    #---------------------------------------------------------------#
                    val_gen_data = model.decode(noise_dist.sample(10000))
                    gen_data = model.decode(noise_dist.sample(100000))
                    print('\nComputing log-likelihood at iter %d'%\
                        (id))
                    self.compute_log_likelihood(val_gen_data, gen_data, id)
                
                if flags.DATASET == 'color_mnist':
                    #---------------------------------------------------------------#
                    # compute FID                                                   #
                    #---------------------------------------------------------------#
                    print('\nComputing FID at iter %d'%(id))
                    path = 'utils/inception_statistics/color_mnist'
                    if not os.path.isdir(path):
                        os.mkdir(path)
                    f = os.path.join(path,'real_statistics')
                    if not os.path.isdir(f + '.npz'):
                        x,_ = data_dist.sample(data_dist.num_train_samples)
                        data = x
                        x,_ = data_dist.test_sample()
                        data = np.hstack((data,x))
                        x,_ = data_dist.val_sample()
                        data = np.hstack((data,x))
                        # reshape data before passing to inception net
                        images = self.reshape_cmnist(data)
                        images = np.round(images*255)
                        real_mu, real_sigma = self.compute_fid(images, 'utils/inception-2015-12-05/classify_image_graph_def.pb')
                        np.savez(f, mu=real_mu, sigma=real_sigma)
                    else:
                        npzfile = np.load(f)
                        real_mu, real_sigma = npzfile['mu'], npzfile['sigma']
                    
                    gen_data = model.decode(noise_dist.sample(10000))
                    # reshape data before passing to inception net
                    images = self.reshape_cmnist(gen_data)
                    images = np.round(images*255)
                    gen_mu, gen_sigma = self.compute_fid(images, 'utils/inception-2015-12-05/classify_image_graph_def.pb')
                    fid_value = fid.calculate_frechet_distance(gen_mu, gen_sigma, real_mu, real_sigma)
                    print("FID: %s" % fid_value)


parser = argparse.ArgumentParser(description='Evaluate generative models')

parser.add_argument('--method', action='store', required=True, help='one of methods: \
                                                        pae (Plummer Autoencoder),\
                                                        vae (Variational Autoencoder),\
                                                        wgan (Wasserstein GAN with Gradient Penalty),\
                                                        veegan (Variational Encoder Enhancement to GAN),\
                                                        cougan (Coulumb GAN),\
                                                        avbac (Adversarial Variational Bayes with Adapt. Contrast),\
                                                        aae (Adversarial Autoencoders),\
                                                        bigan (Bidirectional GAN)')
parser.add_argument('--working_dir', action='store', required=True, help='path to save and load model information')
parser.add_argument('--db', action='store', required=True, help='one of datasets: \
                                                        grid (25 modes uniformly distributed in a rectangle),\
                                                        low_dim_embed (10d gaussians embedded in 1000d space)\
                                                        color_mnist (stacked MNIST)')
parser.add_argument('--hidden_acti', action='store', required=False, default='relu', help='activation functions using in \
                                                        hidden layers: linear, relu, selu, elu, sigmoid, tanh. Default is relu')
parser.add_argument('--enc_out_acti', action='store', required=False, default='linear', help='activation function using in output layer of the encryptor: \
                                                        linear, relu, selu, elu, sigmoid, tanh. Default is linear')
parser.add_argument('--gen_out_acti', action='store', required=False, default='linear', help='activation function using in output layer of the generator: \
                                                        linear, relu, selu, elu, sigmoid, tanh. Default is linear')
parser.add_argument('--disc_out_acti', action='store', required=False, default='linear', help='activation function using in output layer of the discriminator: \
                                                        linear, relu, selu, elu, sigmoid, tanh. Default is linear')
parser.add_argument('--ckpt_id', action='store', nargs='*', type=int, required=True, help='checkpoint id. Multiple ckpt_ids are separated by space')

parser.add_argument('--batch_norm', action='store', type=int, required=False, default=0, help='batch_normalization')

'''
main function
'''
if __name__ == '__main__':
    tf.reset_default_graph()
    from setup import *

    args = parser.parse_args()
    args.train_enc_steps = 0 # no training anymore

    METHOD = args.method
    DATASET = args.db
    flags = SETUP(DATASET)
    noise_dist = UniformNoise(flags.NOISE_DIM)

    if DATASET == 'grid':
        data_dist = Grid()
    elif DATASET == 'low_dim_embed':
        data_dist = LowDimEmbed()
    elif DATASET == 'color_mnist':
        data_dist = CMNIST('data/mnist')
        
    if METHOD == 'pae':
        from pae import PAE
        model = PAE(data_dist, noise_dist, flags, args)
    elif METHOD == 'cougan':
        noise_dist = NormalNoise(flags.NOISE_DIM)
        from coulomb import CoulombGAN
        model = CoulombGAN(data_dist, noise_dist, flags, args)
    elif METHOD == 'bigan':
        from bigan import BiGAN
        model = BiGAN(data_dist, noise_dist, flags, args)
    elif METHOD == 'veegan':
        if DATASET == 'grid':
            flags.NOISE_DIM = 254 # refer to author's implementation
            flags.GEN_ARCH[0] = flags.NOISE_DIM
            flags.ENC_ARCH[-1] = flags.NOISE_DIM
            flags.DISC_ARCH[0] = flags.NOISE_DIM + flags.DATA_DIM
            #flags.DISC_ARCH = [flags.NOISE_DIM+flags.DATA_DIM, 128, 1]
        elif DATASET == 'low_dim_embed':
            flags.DISC_ARCH = [flags.NOISE_DIM+flags.DATA_DIM, 128, 128, 1]
            flags.LEARNING_RATE = 1e-3
        else:
            flags.LEARNING_RATE = 2e-4 # follow radford
            flags.EXPONENTIAL_DECAY = True
        noise_dist = UniformNoise(flags.NOISE_DIM)
        input_noise_dist = UniformNoise(1)
        from veegan import VEEGAN
        model = VEEGAN(data_dist, noise_dist, input_noise_dist, flags, args)
    elif METHOD == 'wgan':
        noise_dist = NormalNoise(flags.NOISE_DIM)
        eps_dist = UniformNoise(1)
        from wgan import WGANGP
        model = WGANGP(data_dist, noise_dist, eps_dist, flags, args)
    elif METHOD == 'vae':
        from vae import VAE
        noise_dist = NormalNoise(flags.NOISE_DIM)
        model = VAE(data_dist, noise_dist, flags, args)
    elif METHOD == 'aae':
        from aae import AAE
        model = AAE(data_dist, noise_dist, flags, args)
    elif METHOD == 'avbac':
        from avbac import *
        noise_dist = NormalNoise(flags.NOISE_DIM)
        single_noise_dist = NormalNoise(PERTURB)
        model = AVB_AC(data_dist, noise_dist, single_noise_dist, flags, args)
    
    model.create_model()

    evaluator = Evaluator(model)
    for id in args.ckpt_id:
        evaluator.run(os.path.join(args.working_dir,'model'), [id])
