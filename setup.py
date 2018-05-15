class SETUP:
    def __init__(self,DATASET):
        self.DATASET = DATASET
        if self.DATASET == 'ring':
            self.NUM_SAMPLES = 800
            self.BATCH_SIZE = self.NUM_SAMPLES
            self.EPOCHS = 1000000
            self.SKIP = 10000
            self.LEARNING_RATE = 0.0001
            self.EXPONENTIAL_DECAY = False
            self.LAMBDA = 1
            self.DATA_DIM = 2
            self.NOISE_DIM = 2
            self.GEN_ARCH = [self.NOISE_DIM, 128, 128, self.DATA_DIM] # Unterthiner et al. 2017
            self.ENC_ARCH = [self.DATA_DIM, 128, 128, self.NOISE_DIM]
            self.DISC_ARCH = [self.NOISE_DIM+self.DATA_DIM, 128, 128, 1]
        elif self.DATASET == 'grid':
            self.NUM_SAMPLES = 500
            self.BATCH_SIZE = self.NUM_SAMPLES
            self.EPOCHS = 2000000
            self.SKIP = 20000
            self.LEARNING_RATE = 1e-4
            self.EXPONENTIAL_DECAY = False
            self.LAMBDA = 1
            self.DATA_DIM = 2
            self.NOISE_DIM = 2
            self.GEN_ARCH = [self.NOISE_DIM, 128, 128, self.DATA_DIM] # Unterthiner et al. 2017
            self.ENC_ARCH = [self.DATA_DIM, 128, 128, self.NOISE_DIM]
            self.DISC_ARCH = [self.NOISE_DIM+self.DATA_DIM, 128, 128, 1]
        elif self.DATASET == 'low_dim_embed':
            self.NUM_SAMPLES = 500
            self.BATCH_SIZE = 500
            self.EPOCHS = 2000000
            self.SKIP = 20000
            self.LEARNING_RATE = 1e-4
            self.EXPONENTIAL_DECAY = False
            self.LAMBDA = 1
            self.DATA_DIM = 1000
            self.NOISE_DIM = 10
            self.GEN_ARCH = [self.NOISE_DIM, 128, 128, self.DATA_DIM] # Unterthiner et al. 2017
            self.ENC_ARCH = [self.DATA_DIM, 128, 128, self.NOISE_DIM]
            self.DISC_ARCH = [self.NOISE_DIM+self.DATA_DIM, 128, 128, 1]
        elif self.DATASET == 'gray_mnist':
            self.NUM_SAMPLES = 50000
            self.EPOCHS = 5000
            self.SKIP = 10000
            self.LEARNING_RATE = 1e-4
            self.LAMBDA = 1
            self.DATA_DIM = 28*28
            self.NOISE_DIM = 10
            self.BATCH_SIZE = 128
            self.GEN_ARCH = [self.NOISE_DIM, 500, 1000, 1500, 2000, 2500, self.DATA_DIM] # Ciresan et al. 2010
            self.ENC_ARCH = [self.DATA_DIM, 2500, 2000, 1500, 1000, 500, self.NOISE_DIM]
            self.DISC_ARCH = [self.NOISE_DIM+self.DATA_DIM, 2500, 2000, 1500, 1000, 500, 1]
        elif self.DATASET == 'color_mnist':
            self.NUM_SAMPLES = 50000
            self.EPOCHS = 4000
            self.SKIP = 130000
            self.LEARNING_RATE = 1e-4
            self.EXPONENTIAL_DECAY = False
            self.LAMBDA = 1
            self.DATA_DIM = 28*28*3
            self.NOISE_DIM = 10
            self.BATCH_SIZE = 128
            self.GEN_ARCH = [self.NOISE_DIM, 500, 1000, 1500, 2000, 2500, self.DATA_DIM] # Ciresan et al. 2010
            self.ENC_ARCH = [self.DATA_DIM, 2500, 2000, 1500, 1000, 500, self.NOISE_DIM]
            self.DISC_ARCH = [self.NOISE_DIM+self.DATA_DIM, 2500, 2000, 1500, 1000, 500, 1]
        elif self.DATASET == 'cifar_100':
            self.NUM_SAMPLES = 50000
            self.EPOCHS = 6000
            self.SKIP = 130000
            self.LEARNING_RATE = 1e-4
            self.EXPONENTIAL_DECAY = False
            self.LAMBDA = 1
            self.DATA_DIM = 32*32*3
            self.BATCH_SIZE = 128
            self.NOISE_DIM = 5
            self.GEN_ARCH = [self.NOISE_DIM, 3072, 3072, 3072, 3072, 3072, self.DATA_DIM]
            self.ENC_ARCH = [self.DATA_DIM, 3072, 3072, 3072, 3072, 3072, self.NOISE_DIM]
            self.DISC_ARCH = [self.NOISE_DIM+self.DATA_DIM, 3072, 3072, 3072, 3072, 3072, 1]