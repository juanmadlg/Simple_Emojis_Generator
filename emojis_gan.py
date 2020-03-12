import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten
from tensorflow.keras.optimizers import Adam


class Generator:
    def __init__(self, img_rows, img_cols, channels, z_dim):
        """
        Returns the Model for the Generator.
        It is a very simple NN that has as an input a random vector and generates an output with img_shape
        :param img_shape: Shape of the images that the Generator generates
        :param z_dim: Size of the random vector that it is used to start the generation
        :return: Generator's Model
        """
        img_shape = (img_rows, img_cols, channels)

        model = Sequential()
        # First layer + activation function
        model.add(Dense(128, input_dim=z_dim))
        model.add(LeakyReLU(alpha=0.01))
        # Output layer
        model.add(Dense(img_rows * img_cols * channels, activation='tanh'))
        # Reshape to the expected image shape
        model.add(Reshape(img_shape))

        self._model = model

    @property
    def model(self):
        return self._model

    def generate(self, batch_size):
        # Creates batch_size random vectors for the Generator
        z = np.random.normal(0, 1, (batch_size, 100))
        return self._model.predict(z)


class Discriminator:
    def __init__(self, img_rows, img_cols, channels):
        """
        Returns the model of the Discriminator
        Another simple NN that has an image as an input
        and a true/false (it is a real image or a fake one) as a output
        :param img_shape: Shape of the input image
        :return: Discriminator Model
        """
        img_shape = (img_rows, img_cols, channels)

        model = Sequential()
        model.add(Flatten(input_shape=img_shape))
        # First layer + Activation Function
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.01))
        # Output layer that returns if the input is real or fake
        model.add(Dense(1, activation='sigmoid'))

        self._model = model

    @property
    def model(self):
        return self._model

    def compile(self):
        self._model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        # The Discriminator is not trained inside the GAN.
        # It is trained individually
        self._model.trainable = False


class TrainingSet:
    def __init__(self, data):
        self._data = data

    def prepare(self):
        """
        Preparing data:
        * Origin values: from 0 to 255. Result values: from -1 to 1
        * Origin shape: (N, 28, 28). Result shape: (N, 28, 28, 1)
        """
        self._data = np.expand_dims(self._data / 127.5 - 1.0, axis=3)
        return self

    def get_random_batch(self, batch_size):
        # Gets batch_size random index
        idx = np.random.randint(0, self._data.shape[0], batch_size)
        # Returns the examples (images)
        return self._data[idx]


class GAN:
    def __init__(self, generator, discriminator):
        """
        GAN as a Secuential Model of a Generator and a Discriminator
        :param generator:
        :param discriminator:
        :return:
        """
        self._generator = generator
        self._discriminator = discriminator

        model = Sequential()
        model.add(generator.model)
        model.add(discriminator.model)

        self._model = model

    @property
    def model(self):
        return self._model

    @property
    def generator(self):
        return self._generator

    @property
    def discriminator(self):
        return self._discriminator

    def compile(self):
        self._model.compile(loss='binary_crossentropy', optimizer=Adam())

    def train(self, x_train, iterations, batch_size, sample_interval):
        losses = []
        accuracies = []
        iteration_checkpoints = []

        x_train = TrainingSet(x_train).prepare()

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for iteration in range(iterations):

            real_images = x_train.get_random_batch(batch_size)
            generated_images = self._generator.generate(batch_size)

            # Trains discriminator (individually, not with the itself)
            d_loss_real = self._discriminator.model.train_on_batch(real_images, real)
            d_loss_fake = self._discriminator.model.train_on_batch(generated_images, fake)
            d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the GAN
            z = np.random.normal(0, 1, (batch_size, 100))
            g_loss = self._model.train_on_batch(z, real)

            if (iteration + 1) % sample_interval == 0:
                # Stores interval measurements
                losses.append((d_loss, g_loss))
                accuracies.append(100.0 * accuracy)
                iteration_checkpoints.append(iteration + 1)

                st.write(f"Iteration: {iteration + 1}:");
                st.text(f"Discriminator loss: {d_loss}, acc.: {100.0 * accuracy}")
                st.text(f"GAN loss: {g_loss}")

                self._sample_images(iteration+1)

    def _sample_images(self, iteration):
        # Prepares 5 random vectors
        z = np.random.normal(0, 1, (8, 100))

        gen_imgs = self._generator.model.predict(z)

        # Transforms the data to be images
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = gen_imgs * 255

        index = 0
        images = []
        for image in gen_imgs:
            file_name = f"output/{iteration}_{index}.png"
            cv2.imwrite(file_name, image[:, :, 0])
            images.append(Image.open(file_name))
            index += 1

        st.image(images)
