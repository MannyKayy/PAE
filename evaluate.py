from __future__ import division
import argparse
import os
from data import *
import tensorflow as tf
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn import mixture
from sklearn.model_selection import cross_val_score, GridSearchCV
from classifier import *
import fid
import itertools
import platform
import matplotlib
import matplotlib.pyplot as plt
if 'Linux' in platform.platform():
    plt.switch_backend('agg')

NOISE_PERTURB = 0.005
SEED = 100

class Evaluator:
    def __init__(self, model):
        self.model = model
    
    def visualize(self, gen_data, ckpt_id, real_data=None):
        '''
            This function visualizes data
            Each distribution has it own data visualization
        '''
        working_dir = self.model.working_dir
        data_dist.visualize(gen_data, real_data, save_path=os.path.join(working_dir,'figure',DATASET + '-' + str(ckpt_id) + '.png'))
    
    def visualize_closest(self, test_data, gen_data, ckpt_id):
        '''
            This function visualizes generated data closest to real data
            Each distribution has it own data visualization
        '''
        working_dir = self.model.working_dir
        data_dist.visualize(gen_data, test_data, save_path=os.path.join(working_dir,'figure',  DATASET + '-nr-' + str(ckpt_id) + '.png'))

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

    def compute_frechet_distance_mat(self, gen_data, ckpt_id):
        if flags.DATASET == 'grid':
            n_components = 25
        elif flags.DATASET == 'low_dim_embed':
            n_components = 10
        else:
            print('The dataset is not supported for frechet distance computation')
            return
        # fit mixture of gaussians on real data
        test_data,_ = data_dist.test_sample(gen_data.shape[1])
        gm_true = mixture.GaussianMixture(n_components, covariance_type='full')
        gm_true.fit(test_data.T)
        mu_true = gm_true.means_
        cov_true = gm_true.covariances_
        print('positive definite: {0}'.format(self.is_pos_def(cov_true)))
        # fit mixture of gaussians on generated data
        test_data,_ = data_dist.test_sample(gen_data.shape[1])
        gm_fake = mixture.GaussianMixture(n_components, covariance_type='full')
        gm_fake.fit(test_data.T)
        mu_fake = gm_fake.means_
        cov_fake = gm_fake.covariances_
        print('positive definite: {0}'.format(self.is_pos_def(cov_fake)))
        # compute Frechet distance matrix
        distances = np.zeros((n_components,n_components))
        for i in range(n_components):
            #print('mode mean {0}: {1}'.format(i+1,np.max(mu_fake[i,:])))
            for j in range(n_components):
                v = calculate_frechet_distance(mu_true[i,:], cov_true[i,:,:], mu_fake[j,:], cov_fake[j,:,:])
                # if np.iscomplex(v).any():
                #     print(v)
                # else:
                distances[i,j] = v
        # show the matrix
        distances = np.around(distances, 2)
        min_distances = np.min(distances, axis=1).tolist()
        closest_modes = np.argmin(distances, axis=1)+1
        tick_marks = [i+1 for i in range(n_components)]
        plt.plot(tick_marks, min_distances, 'b^')
        for i,j in zip(range(n_components),min_distances):
            plt.text(i+1,j,str(closest_modes[i]), color='red')
        plt.xticks(tick_marks)
        plt.ylabel('Distance to closest mode')
        #plt.ylim(0,1)
        plt.xlim(0,n_components+1)
        plt.xlabel('Mode of real data')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def reshape_data(self, x, db):
        '''
            This functions reshape data so that it is compatible for CNN
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

    def reshape_cifar(self, data):
        # reshape data to [batch_size, 32, 32, 3]
        data_new = np.zeros((data.shape[1],32,32,3))
        for i in range(data.shape[1]):
            for c in range(3):
                data_new[i,:,:,c] = data[c*32*32:(c+1)*32*32,i].reshape(32,32)
        return data_new

    def run(self, model_path, ckpt_ids):
        '''
            This is the main function for evaluation
            Input:
                model_path: the folder containing all checkpoints
                ckpt_ids: the list of integers indicating checkpoint ids
                that we want to evaluate
        '''
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        model = self.model
        with tf.Session(config=config) as sess:
            # start evaluating every checkpoint
            for id in ckpt_ids:     
                ckpt = os.path.join(model_path, 'saved-model-'+str(id))
                saver = tf.train.Saver()
                saver.restore(sess, ckpt)
                
                if flags.DATASET in ['grid']:
                    #---------------------------------------------------------------#
                    # visualization real and random generated samples               #
                    #---------------------------------------------------------------#
                    print('\nVisualizing generated samples at iter %d'%(id))
                    gen_data = model.decode(noise_dist.sample(1000))
                    self.visualize(gen_data,id)

                if flags.DATASET in ['color_mnist']:
                    #---------------------------------------------------------------#
                    # visualization real and random generated samples               #
                    #---------------------------------------------------------------#
                    np.random.seed(SEED)
                    print('\nVisualizing generated samples at iter %d'%(id))
                    real_data,_ = data_dist.sample(10)
                    gen_data = model.decode(noise_dist.sample(model.batch_size))
                    self.visualize(gen_data,id,real_data)
                    
                if flags.DATASET == 'cifar_100':
                    #---------------------------------------------------------------#
                    # visualization real and random generated samples               #
                    #---------------------------------------------------------------#
                    np.random.seed(SEED)
                    print('\nVisualizing generated samples at iter %d'%(id))
                    gen_data = model.decode(noise_dist.sample(model.batch_size))
                    self.visualize(gen_data,id)
                
                if flags.DATASET in ['color_mnist', 'cifar_100'] and args.method not in ['cougan','wgan']:
                    #---------------------------------------------------------------#
                    # visualization real and nearest generated samples              #
                    #---------------------------------------------------------------#
                    np.random.seed(SEED)
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
                        perturbed_input = np.random.multivariate_normal(z_sampled[:,i],NOISE_PERTURB*np.eye(z_sampled.shape[0]),9)
                        gen_data[:,i*9:i*9+9] = model.decode(perturbed_input.T)
                    self.visualize_closest(real_data,gen_data,id)
                
                log = {}
                if flags.DATASET in ['low_dim_embed']:
                    #---------------------------------------------------------------#
                    # counting the number of modes using pretrained MLP net          #
                    #---------------------------------------------------------------#
                    print('\nClassifying generated data at iter %d'%(id))
                    gen_data = model.decode(noise_dist.sample(10000))
                    predictions = self.classify_low_dim_embed(os.path.join('classification','low_dim_embed','model'), gen_data)
                    hist,_ = np.histogram(predictions)
                    log['hist'] = hist
                    if not os.path.exists(os.path.join(args.working_dir,'others')):
                        os.makedirs(os.path.join(args.working_dir,'others'))
                    path = os.path.join(args.working_dir,'others','hist_modes.pkl')
                    with open(path,'wb') as f:
                        pickle.dump(log,f)

                if flags.DATASET in ['color_mnist']:
                    #---------------------------------------------------------------#
                    # counting the number of modes using pretrained CNN          #
                    #---------------------------------------------------------------#
                    print('\nClassifying generated data at iter %d'%(id))
                    hist = 0
                    for i in range(100):
                        gen_data = model.decode(noise_dist.sample(10000))
                        predictions = self.classify_cmnist(os.path.join('classification','color_mnist','model'), gen_data)
                        h,_ = np.histogram(predictions, bins=1000)
                        hist = hist + h
                    print('number of class having 0 samples %d'%np.sum(hist==0))
                    print('mean #samples per class %d'%np.mean(hist))
                    print('std #samples per class %d'%np.std(hist))
                    log['hist'] = hist
                    if not os.path.exists(os.path.join(args.working_dir,'others')):
                        os.makedirs(os.path.join(args.working_dir,'others'))
                    path = os.path.join(args.working_dir,'others','hist_modes.pkl')
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
                
                if flags.DATASET in ['color_mnist']:
                    #---------------------------------------------------------------#
                    # compute FID                                                   #
                    #---------------------------------------------------------------#
                    print('\nComputing FID at iter %d'%(id))
                    path = os.path.join('utils','inception_statistics','color_mnist')
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
                        real_mu, real_sigma = self.compute_fid(images, os.path.join('utils','inception-2015-12-05','classify_image_graph_def.pb'))
                        np.savez(f, mu=real_mu, sigma=real_sigma)
                    else:
                        npzfile = np.load(f)
                        real_mu, real_sigma = npzfile['mu'], npzfile['sigma']
                    
                    gen_data = model.decode(noise_dist.sample(10000))
                    # reshape data before passing to inception net
                    images = self.reshape_cmnist(gen_data)
                    images = np.round(images*255)
                    gen_mu, gen_sigma = self.compute_fid(images, os.path.join('utils','inception-2015-12-05','classify_image_graph_def.pb'))
                    fid_value = fid.calculate_frechet_distance(gen_mu, gen_sigma, real_mu, real_sigma)
                    print("FID: %s" % fid_value)

                if flags.DATASET in ['cifar_100']:
                    #---------------------------------------------------------------#
                    # compute FID                                                   #
                    #---------------------------------------------------------------#
                    print('\nComputing FID at iter %d'%(id))
                    path = os.path.join('utils','inception_statistics','cifar_100')
                    if not os.path.isdir(path):
                        os.mkdir(path)
                    f = os.path.join(path,'real_statistics')
                    if not os.path.isdir(f + '.npz'):
                        x,_ = data_dist.sample(data_dist.num_train_samples)
                        data = x
                        x,_ = data_dist.test_sample()
                        data = np.hstack((data,x))
                        # reshape data before passing to inception net
                        images = self.reshape_cifar(data)
                        images = np.round(images*255)
                        real_mu, real_sigma = self.compute_fid(images, os.path.join('utils','inception-2015-12-05','classify_image_graph_def.pb'))
                        np.savez(f, mu=real_mu, sigma=real_sigma)
                    else:
                        npzfile = np.load(f)
                        real_mu, real_sigma = npzfile['mu'], npzfile['sigma']
                    
                    gen_data = model.decode(noise_dist.sample(10000))
                    # reshape data before passing to inception net
                    images = self.reshape_cifar(gen_data)
                    images = np.round(images*255)
                    gen_mu, gen_sigma = self.compute_fid(images, os.path.join('utils','inception-2015-12-05','classify_image_graph_def.pb'))
                    fid_value = fid.calculate_frechet_distance(gen_mu, gen_sigma, real_mu, real_sigma)
                    print("FID: %s" % fid_value)


parser = argparse.ArgumentParser(description='Evaluate generative models')

parser.add_argument('--method', action='store', required=True, help='one of methods: \
                                                        pae (Plummer Autoencoder),\
                                                        wae (Wasserstein Autoencoder),\
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
                                                        color_mnist (stacked MNIST),\
                                                        cifar_100 (CIFAR 100 categories)')
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

parser.add_argument('--gpu', action='store', required=False, type=int, default=0, help='selection of GPU device (integer starting from 0)')


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
        data_dist = CMNIST(os.path.join('data','mnist'))
    elif DATASET == 'cifar_100':
        data_dist = CIFAR100(os.path.join('data','cifar-100'))
    if METHOD == 'pae':
        from pae import PAE
        model = PAE(data_dist, noise_dist, flags, args)
    elif METHOD == 'wae':
        from wae import WAE
        model = WAE(data_dist, noise_dist, flags, args)
    elif METHOD == 'cougan':
        noise_dist = NormalNoise(flags.NOISE_DIM)
        from cougan import CoulombGAN
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
    evaluator.run(os.path.join(args.working_dir,'model'), args.ckpt_id)
