import os
import cv2
import numpy as np


# Training dataset - Images 28x28
# The training dataset determines the knd of examples the Generator will learn to emulate.
from emojis_gan import Discriminator, Generator, GAN


# Helper function to read the Training Set (images) from a folder
def load_images(folder):
    instances = []

    # Load in the images
    for file_path in os.listdir(f'{folder}/'):
        instances.append(cv2.imread(f'{folder}/{file_path}', 0))

    return np.array(instances)


img_rows = 28
img_cols = 28
channels = 1

# Size of the noise vector
z_dim = 100

# Creating and compiling the Discriminator
discriminator = Discriminator(img_rows, img_cols, channels)
discriminator.compile()
# Creating the Generator
generator = Generator(img_rows, img_cols, channels, z_dim)
# Creating the GAN and compiling it.
gan = GAN(generator, discriminator)
gan.compile()

# Hyper-parameters for the Training
iterations = 50000
batch_size = 128 # Important in terms of memory usage
# Interval to get training info an get current examples
sample_interval = 1000

# Training
gan.train(load_images('data'), iterations, batch_size, sample_interval)
