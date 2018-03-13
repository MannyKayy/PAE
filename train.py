import argparse
import os

parser = argparse.ArgumentParser(description='Train generative models')

parser.add_argument('--method', action='store', required=True, help='one of methods: \
                                                        pae (Plummner Autoencoder),\
                                                        vae (Variational Autoencoder),\
                                                        wgan (Wasserstein GAN with Gradient Penalty),\
                                                        veegan (Variational Encoder Enhancement to GAN),\
                                                        cougan (Coulumb GAN),\
                                                        bigan (Bidirectional GAN)')
parser.add_argument('--working_dir', action='store', required=True, help='path to save and load model information')
parser.add_argument('--db', action='store', required=True, help='one of datasets: \
                                                        grid (25 modes uniformly distributed in a rectangle),\
                                                        low_dim_embed (10d gaussians embedded in 1000d space),\
                                                        color_mnist (stacked MNIST)')
                                                        
parser.add_argument('--hidden_acti', action='store', required=False, default='relu', help='activation functions using in \
                                                        hidden layers: linear, relu, selu, elu, sigmoid, tanh. Default is relu')
parser.add_argument('--enc_out_acti', action='store', required=False, default='linear', help='activation function using in output layer of the encryptor: \
                                                        linear, relu, selu, elu, sigmoid, tanh. Default is linear')
parser.add_argument('--gen_out_acti', action='store', required=False, default='linear', help='activation function using in output layer of the generator: \
                                                        linear, relu, selu, elu, sigmoid, tanh. Default is linear')
parser.add_argument('--disc_out_acti', action='store', required=False, default='linear', help='activation function using in output layer of the discriminator: \
                                                        linear, relu, selu, elu, sigmoid, tanh. Default is linear')
parser.add_argument('--batch_norm', action='store', required=False, type=int, default=0, help='batch normalization')

'''
main function
'''
if __name__ == '__main__':
    from setup import *
    args = parser.parse_args()
    WORKING_DIR = args.working_dir
    if not os.path.isdir(args.method):
        os.mkdir(args.method)
    if not os.path.isdir(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    if not os.path.isdir(os.path.join(WORKING_DIR,'model')):
        os.mkdir(os.path.join(WORKING_DIR,'model'))
    if not os.path.isdir(os.path.join(WORKING_DIR,'figure')):
            os.mkdir(os.path.join(WORKING_DIR,'figure'))
    
    METHOD = args.method
    if METHOD == 'pae':
        import pae
        pae.run(args)
    elif METHOD == 'cougan':
        import cougan
        cougan.run(args)
    elif METHOD == 'bigan':
        import bigan
        bigan.run(args)
    elif METHOD == 'veegan':
        import veegan
        veegan.run(args)
    elif METHOD == 'wgan':
        import wgan
        wgan.run(args)
    elif METHOD == 'aae':
        import aae
        aae.run(args)
    elif METHOD == 'avbac':
        import avbac
        avbac.run(args)
    elif METHOD == 'vae':
        import vae
        vae.run(args)
    
    