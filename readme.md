
# Plummer AutoEncoder (PAE)
This repository contains the Tensorflow implementation of paper [Plummer Autoencoders](https://arxiv.org/abs/1802.03505).
We provide the implementation of the following methods to guarantee that all results are reproducible:

1. Plummer AutoEncoder (pae)
2. Wasserstein GAN with Gradient Penalty (wgan)
3. Coulomb GAN (cougan)
4. Bidirectional GAN (bigan)
5. Variational Encoder Enhancement to GAN (veegan)
6. Variational AutoEncoder (vae)
7. Adversarial AutoEncoder (aae)
8. Adversarial Variational Bayes with Adaptive Contrast (avbac)

### Animation on Stacked MNIST

We generate samples during the training of PAE.

<img src='animations/cmnist.gif' width=400px>

### How to run the code

1. **Setup**

*setup.py* contains the general settings for different *datasets* (e.g. the architecture of generator, discriminator and encoder). 
Supported datasets are:

* **grid**: 25 Gaussians distributed uniformly on a grid.
* **low_dim_embed**: 10 10d isotropic Gaussians embedded into 1000d space.
* **color_mnist**: MNIST digits stacked on top of each other to create color digits.

2. **Training**

 Use *train.py*. The following parameters need to be set:

* *--method*: indicating one of the supported algorithms. This parameter is *required*.
* *--db*: indicating one of the supported datasets. This parameter is *required*.
* *--working_dir*: indicating the folder to store all training information. This parameter is *required*. We can run multiple training instances in parallel with different working dirs.
* *--hidden_acti*: activation function of hidden units. It is applied to both of generator, discriminator, and encoder. Currently *linear, relu, selu, sigmoid, tanh* are supported. This parameter is *optional*, and *linear* is applied by default.
* *--gen_out_acti*: activation function of output units in the generator. Currently *linear, relu, selu, sigmoid, tanh* are supported. This parameter is *optional*, and *linear* is applied by default.
* *--disc_out_acti*: activation function of output units in the discriminator. Currently *linear, relu, selu, sigmoid, tanh* are supported. This parameter is *optional*, and *linear* is applied by default.
* *--enc_out_acti*: activation function of output units in the discriminator. Currently *linear, relu, selu, sigmoid, tanh* are supported. This parameter is *optional*, and *linear* is applied by default.
* *--batch_norm*: batch normalization. This parameter is *optional*, and *0* is applied by default.

Examples:

* Train pae on CMNIST:

```
python train.py --method pae --db color_mnist --working_dir pae/color_mnist --gen_out_acti sigmoid
```

* Train cougan on grid:

```
python train.py --method cougan --db grid --working_dir cougan/grid
```

* Train cougan on grid with batch normalization:

```
python train.py --method cougan --db grid --batch_norm 1 --working_dir cougan/grid
```

3. **Evaluation**

Before running evaluation (on low_dim_embed or CMNIST), you have to train the classifier which is used later to detect mode collapse.

Examples:

* Train classifier on low_dim_embed:

```
python classifier.py --db low_dim_embed --working_dir classification/low_dim_embed
```

* Train classifier on CMNIST:

```
python classifier.py --db color_mnist --working_dir classification/color_mnist
```

In *evaluate.py*, you can find the code for evaluation. This file is called with similar parameters as *training.py* except:

* *--ckpt_id*: indicating checkpoint ids of model that we want to evaluate. Multiple ids are supported.

Examples:

* Evaluate pae trained on color_mnist at checkpoint id 1001 and 2001:

```
python evaluate.py --method pae --db color_mnist --working_dir pae/color_mnist --gen_out_acti sigmoid --ckpt_id 1001 2001
```

### More information
1. Type: ```python train.py --help```
2. Contact us
