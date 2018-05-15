from __future__ import division
import numpy as np
import pickle
import os
import sys
import tarfile
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 
font = {'family' : "serif",
        'size'   : 15}
matplotlib.rc('font', **font)
import scipy.misc
import time

# Training data
class UniformNoise:
    def __init__(self, dim):
        self.dim = dim

    def sample(self, N):
        samples = np.random.uniform(-1.0, 1.0, (self.dim, N))
        return samples

class NormalNoise:
    def __init__(self, dim):
        self.dim = dim

    def sample(self, N):
        samples = np.random.normal(0.0, 1.0, (self.dim, N))
        return samples

class Unimodal:
    def __init__(self, dim):
        self.dim = dim

    def sample(self, N):
        samples = np.random.normal(4.0, 0.5, (self.dim, N))
        return samples

    def shuffle(self, samples):
        _, N = np.shape(samples)
        id = np.random.permutation(N)
        samples = samples[:,id]
        return samples

class Bimodal:
    def __init__(self, dim):
        self.dim = dim

    def sample(self, N):
        samples1 = np.random.normal(4.0, 0.5, (self.dim, int(N/2)))
        samples2 = np.random.normal(-4.0, 0.5, (self.dim, int(N/2)))
        samples = np.hstack((samples1,samples2))
        return samples

    def shuffle(self, samples):
        _, N = np.shape(samples)
        id = np.random.permutation(N)
        samples = samples[:,id]
        return samples

class Ring:
    def __init__(self):
        self.modes = 8

    def sample(self, N):
        cov = [[0.05, 0.0],[0.0, 0.05]]
        for i in range(self.modes):
            angle = 2*i*np.pi/self.modes
            radius = 2.0
            mean = [radius*np.cos(angle), radius*np.sin(angle)]
            sample = np.random.multivariate_normal(mean, cov, int(N/self.modes)).T
            if i == 0:
                samples = sample
            else:
                samples = np.hstack((samples,sample))
        return samples

    def shuffle(self, samples):
        _, N = np.shape(samples)
        id = np.random.permutation(N)
        samples = samples[:,id]
        return samples

class Grid:
    def __init__(self):
        self.modes = 25

    def sample(self, N):
        cov = [[0.1, 0.0],[0.0, 0.1]]
        num_modes_per_axis = int(np.sqrt(self.modes))
        mode_range = np.linspace(-21,21,num_modes_per_axis)
        x, y = np.meshgrid(mode_range,mode_range)
        x = x.flatten()
        y = y.flatten()
        for i in range(self.modes):
            mean = [x[i], y[i]]
            sample = np.random.multivariate_normal(mean, cov, int(N/self.modes)).T
            if i == 0:
                samples = sample
            else:
                samples = np.hstack((samples,sample))
        return samples, None
    
    def test_sample(self, N):
        return self.sample(N)
        
    def shuffle(self, samples):
        _, N = np.shape(samples)
        id = np.random.permutation(N)
        samples = samples[:,id]
        return samples

    def visualize(self, samples, save_path=None):
        #real_samples,_ = self.sample(gen_samples.shape[1])
        #plt.plot(real_samples[0,:], real_samples[1,:], '.r', label='Real')
        plt.plot(samples[0,:], samples[1,:], '.b', label='Generated')
        plt.xlim((-30, 30))
        plt.ylim((-30, 30))
        #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #   ncol=2, mode="expand", borderaxespad=0.)
        if save_path:
            plt.savefig(save_path,bbox_inches='tight',dpi=800)
            plt.close()
        else:
            plt.show()
            plt.close()

class LowDimEmbed:
    def __init__(self):
        self.hd = 1000
        self.ld = 10
        self.modes = 10
        self.mode_ranges = np.linspace(-5,5,self.modes)
        self.dim_offsets = np.floor(np.linspace(0,self.hd-self.ld-1,self.modes)).astype(int)
    
    def sample(self, batch_size):
        '''
            This function generates a mixture of 10 isotropic Gaussians in 10-d space
            and embeds into 1000-d space
        '''
        for i in range(self.modes):
            mean = np.ones(self.ld)*self.mode_ranges[i]
            cov = np.eye(self.ld)
            sample = np.zeros((self.hd,int(batch_size/self.modes)))#np.random.normal(0,1,(self.hd,int(batch_size/self.modes)))
            offset = self.dim_offsets[i]
            sample[offset:offset+self.ld,:] = np.random.multivariate_normal(mean, cov, int(batch_size/self.modes)).T
            if i == 0:
                samples = sample
                labels = np.ones(int(batch_size/self.modes),dtype=np.int32)*i
            else:
                samples = np.hstack((samples,sample))
                labels = np.concatenate((labels,np.ones(int(batch_size/self.modes),dtype=np.int32)*i))
        return samples, labels

    def test_sample(self, batch_size):
        return self.sample(batch_size)

    def val_sample(self, batch_size):
        return self.sample(batch_size)

class MNIST:
    '''
    This class reads samples from grayscale MNIST
    Usage: 
        mnist = MNIST('data/mnist')
        x_train,y_train = mnist.sample(128)     # x_train.shape = [d,n]
        x_test,y_test = mnist.test_samples()    # x_test.shape = [d,n]
    '''
    def __init__(self, db_path):
        self.mnist = input_data.read_data_sets(db_path, one_hot=True)
        self.num_train_samples = self.mnist.train.num_examples
    def sample(self, batch_size):
        x_batch, y_batch = self.mnist.train.next_batch(batch_size)
        return x_batch.T, y_batch
    def test_sample(self):
        x, y = self.mnist.test.images, self.mnist.test.labels
        return x.T, y
    def val_sample(self):
        x, y = self.mnist.validation.images, self.mnist.validation.labels
    

class CMNIST:
    '''
    This class generate stacked MNIST of 1000 classes
    Usage: 
        cmnist = CMNIST('data/mnist')
        x_train,y_train = cmnist.sample(128)     # x_train.shape = [d,n]
        x_test,y_test = cmnist.test_samples()    # x_test.shape = [d,n]
    '''
    def __init__(self, db_path):
        self.mnist = input_data.read_data_sets(db_path, one_hot=False)
        self.num_train_samples = self.mnist.train.num_examples
        self.num_test_samples = self.mnist.test.num_examples
        self.num_val_samples = self.mnist.validation.num_examples

    def sample(self, batch_size):
        x_batch_color = np.zeros((28*28*3,batch_size),dtype=np.float32)
        y_batch = np.zeros(batch_size,dtype=np.int32)
        for i in range(3):
            x_batch, labels_ = self.mnist.train.next_batch(batch_size)
            x_batch = x_batch.T
            y_batch += labels_.astype(np.int32) * (10**i)
            x_batch_color[i*28*28:(i+1)*28*28,:] = x_batch
        return x_batch_color, y_batch

    def test_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.num_test_samples
        x_color = np.zeros((28*28*3,batch_size),dtype=np.float32)
        y = np.zeros(batch_size,dtype=np.int32)
        x, t = self.mnist.test.images, self.mnist.test.labels
        x = x.T
        rand_inds = np.arange(batch_size)
        for i in range(3):
            np.random.shuffle(rand_inds)
            x_color[i*28*28:(i+1)*28*28,:] = x[:,rand_inds]
            y += t[rand_inds].astype(np.int32) * (10**i)
        return x_color, y

    def val_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.num_val_samples
        x_color = np.zeros((28*28*3,batch_size),dtype=np.float32)
        y = np.zeros(batch_size,dtype=np.int32)
        x, t = self.mnist.validation.images, self.mnist.validation.labels
        x = x.T
        rand_inds = np.arange(batch_size)
        for i in range(3):
            np.random.shuffle(rand_inds)
            x_color[i*28*28:(i+1)*28*28,:] = x[:,rand_inds]
            y += t[rand_inds].astype(np.int32) * (10**i)
        return x_color, y
    
    def visualize(self, gen_data, real_data=None, save_path=None):
        assert gen_data.shape[1] >= 90, 'At least 90 generated images are needed'
        data = np.zeros((28*10,28*10,3))
        for r in range(10):
            for c in range(10):
                for i in range(3):
                    if real_data is None:
                        data[r*28:(r+1)*28,c*28:(c+1)*28,i] = gen_data[i*28*28:(i+1)*28*28,r*10+c].reshape((28,28))
                        continue
                    if c == 0:
                            data[r*28:(r+1)*28,c*28:(c+1)*28,i] = real_data[i*28*28:(i+1)*28*28,r].reshape((28,28))
                    else:
                            data[r*28:(r+1)*28,c*28:(c+1)*28,i] = gen_data[i*28*28:(i+1)*28*28,r*9+c-1].reshape((28,28))
                        
        if save_path:
            plt.imsave(save_path, data, dpi=800)
            plt.close()
        else:
            plt.imshow(data)
            plt.close()

class CIFAR100:
    def __init__(self, db_path):
        self.num_train_samples = 100*500
        self.num_test_samples = 100*100
        self.input_dim = 3072

        # read train data
        with open(os.path.join(db_path,'train'), 'rb') as f:
            if sys.version_info < (3,0):
                train = pickle.load(f) # [self.num_train_samples, self.input_dim]
            else:
                train = pickle.load(f, encoding='latin1') # [self.num_train_samples, self.input_dim]
        self.train_data = train['data'].reshape(train['data'].shape[0], 3, 32, 32)
        self.train_data = self.train_data.transpose((2,3,1,0))
        self.train_data = self.train_data.reshape((-1,self.train_data.shape[3]))
        self.train_coarse_labels = np.array(train['coarse_labels'],dtype=np.int32)
        self.train_labels = np.array(train['fine_labels'],dtype=np.int32)
        
        # convert train data to float32 [0-1]
        self.train_data = self.train_data / 255

        # read test data
        with open(os.path.join(db_path,'test'), 'rb') as f:
            if sys.version_info < (3,0):
                test = pickle.load(f) # [self.num_test_samples, self.input_dim]
            else:
                test = pickle.load(f, encoding='latin1') # [self.num_test_samples, self.input_dim]
        self.test_data = test['data'].reshape(test['data'].shape[0], 3, 32, 32)
        self.test_data = self.test_data.transpose((2,3,1,0))
        self.test_data = self.test_data.reshape((-1,self.test_data.shape[3]))
        self.test_coarse_labels = np.array(test['coarse_labels'],dtype=np.int32)
        self.test_labels = np.array(test['fine_labels'],dtype=np.int32)

        # convert test data to float32 [0-1]
        self.test_data = self.test_data / 255
        
        # indices
        self.indices = np.arange(self.num_train_samples)

    def sample(self,batch_size):
        # shuffle indices
        np.random.shuffle(self.indices)
        samples = self.train_data[:,self.indices[:batch_size]]
        labels = self.train_labels[self.indices[:batch_size]]
        return samples, labels
    
    def test_sample(self,batch_size=None,shuffle=True):
        if batch_size is None:
            batch_size = self.num_test_samples
        indices = np.arange(self.num_test_samples)
        np.random.shuffle(indices)
        return self.test_data[:,indices[:batch_size]], self.test_labels[indices[:batch_size]]

    def visualize(self, gen_data, real_data=None, save_path=None):
        assert gen_data.shape[1] >= 90, 'At least 90 generated images are needed'
        num_img = 10
        res = 32
        sep = 1
        data = np.zeros((res*num_img+sep*(num_img+1),res*num_img+sep*(num_img+1),3))
        for r in range(num_img):
            for c in range(num_img):
                    if real_data is None:
                        r*res
                        data[r*res+sep*(1+r):(r+1)*res+sep*(1+r),c*res+sep*(1+c):(c+1)*res+sep*(1+c),:] = gen_data[:,r*num_img+c].reshape((res,res,3))
                        continue
                    if c == 0:
                        data[r*res+sep*(1+r):(r+1)*res+sep*(1+r),c*res+sep*(1+c):(c+1)*res+sep*(1+c),:] = real_data[:,r].reshape((res,res,3))
                    else:
                        data[r*res+sep*(1+r):(r+1)*res+sep*(1+r),c*res+sep*(1+c):(c+1)*res+sep*(1+c),:] = gen_data[:,r*(num_img-1)+c-1].reshape((res,res,3))
        if save_path:
            plt.imsave(save_path, data, dpi=800)
            plt.close()
        else:
            plt.imshow(data)
            plt.close()
