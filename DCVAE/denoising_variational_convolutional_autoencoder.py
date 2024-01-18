# OWNER : SUBINOY ADHIKARI           
# EMAIL : subinoyadhikari@tifrh.res.in

import sys
import os
import pickle
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Input, Dense, BatchNormalization, \
    Activation, Conv2D, Flatten, Reshape, Conv2DTranspose, MaxPooling2D, Lambda




# --------------------------------VARIATIONAL_CONVOLUTIONAL_AUTOENCODER_LOSS--------------------------------#
def _reconstruction_loss(target_value, predicted_value):  # This is the Mean Squared Error #
    error = target_value - predicted_value
    rec_loss = K.mean(K.square(error),
                      axis=[1, 2, 3])  # axis 0=Batch, 1=width, 2=height, 3=no_of_channels
    return rec_loss

def _kl_loss(model):
    # wrap `_kl_loss` such that it takes the model as an argument,
    # returns a function which can take arbitrary number of arguments
    # (for compatibility with `metrics` and utility in the loss function)
    # and returns the kl loss
    def _cal_kl_loss(*args):
        kl_loss = (-0.5) * K.sum(1 + model.lv - K.square(model.mu) - K.exp(model.lv), axis=1)
    # axis=0 is the batch index, whereas axis=1 is the index over the latent space dimensions chosen for self.mu and self.lv.
    #  So, K.sum(axis=1) is just summing the '1 + log(sig^2) - mu^2 - sig^2) expression over the latent space dimensions for
    #  each specific image, while keeping the values of the different images within the batch separate. #
        return kl_loss
    return _cal_kl_loss



class variational_convolutional_autoencoder:
    """
    This is  DCVAE.
    """

    def __init__(self,
                 input_shape,
                 dense_encoder_neurons,
                 dense_encoder_activation,
                 dense_encoder_wt_initialization,
                 dense_latent_neurons,
                 dense_latent_activation,
                 dense_latent_wt_initialization,
                 dense_decoder_neurons,
                 dense_decoder_activation,
                 dense_decoder_wt_initialization,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 conv_padding,
                 conv_activation,
                 conv_wt_initialization,
                 pool_size,
                 pool_strides,
                 pool_padding,
                 rec_loss_wt,
                 beta,
                 deep_dense,
                 maxpooling):

        self.input_shape=input_shape
        self.encoder_neurons=dense_encoder_neurons
        self.encoder_activation=dense_encoder_activation
        self.encoder_wt_initialization=dense_encoder_wt_initialization
        self.latent_neurons=dense_latent_neurons
        self.latent_activation=dense_latent_activation
        self.latent_wt_initialization=dense_latent_wt_initialization
        self.decoder_neurons=dense_decoder_neurons
        self.decoder_activation=dense_decoder_activation
        self.decoder_wt_initialization=dense_decoder_wt_initialization
        self.conv_filters=conv_filters
        self.conv_kernels=conv_kernels
        self.conv_strides=conv_strides
        self.conv_padding=conv_padding
        self.conv_activation=conv_activation
        self.conv_wt_initialization=conv_wt_initialization
        self.pool_size=pool_size
        self.pool_strides=pool_strides
        self.pool_padding=pool_padding
        self.rec_loss_wt=rec_loss_wt
        self.beta=beta
        self.deep_dense=deep_dense
        self.maxpooling=maxpooling


        # Number of layers in the encoder and decoder
        self._num_encoder_layers=len(dense_encoder_neurons)
        self._num_decoder_layers=len(dense_decoder_neurons)
        self._num_conv_layers=len(conv_filters)
        self._shape_before_dense=None

        self.encoder=None
        self.decoder=None
        self.autoencoder=None
        self.encoder_input=None

        # Call the convolutional autoencoder model
        self._convolutional_model()

    # The convolutional autoencoder model
    def _convolutional_model(self):
        self._encoder_model()
        self._decoder_model()
        self._autoencoder_model()

    # --------------------------------ENCODER--------------------------------#

    def _encoder_model(self):
        enc_inp = self._encoder_input()  # Input layer of the encoder #
        enc_lyr = self._encoder_layers(enc_inp)  # Stacking of the hidden layers of the encoder, excluding the latent layer #
        lec = self._latent_layer(enc_lyr)  # Latent layer #
        self.encoder = Model(inputs=enc_inp,
                             outputs=lec,
                             name="ENCODER")  # Encoder #
        self.encoder_input = enc_inp

    # Encoder input layer #
    def _encoder_input(self):
        enc_inp = Input(shape=self.input_shape,
                        name="ENCODER_INPUT")
        return enc_inp

    # Stacking the hidden layers of the encoder (CONVOLUTIONAL + DENSE) #
    def _encoder_layers(self, enc_inp):

        # Convolutional NN #
        enc = enc_inp
        enc = BatchNormalization(name="BATCH_NORMALIZATION_FOR_THE_INPUT_LAYER_OF_THE_ENCODER")(enc)
        for index in range(self._num_conv_layers):
            enc = Conv2D(filters=self.conv_filters[index],
                         kernel_size=self.conv_kernels[index],
                         strides=self.conv_strides[index],
                         padding=self.conv_padding[index],
                         kernel_initializer=self.conv_wt_initialization[index],
                         use_bias = False,
                         name = f"CONVOLUTIONAL_ENCODER_LAYER_{index + 1}")(enc)

            if (self.maxpooling):
                enc = MaxPooling2D(pool_size=self.pool_size[index],
                                   strides=self.pool_strides[index],
                                   padding=self.pool_padding[index])(enc)

            enc = BatchNormalization(name=f"CONVOLUTIONAL_BATCH_NORMALIZATION_FOR_ENCODER_LAYER_{index + 1}")(enc)
            enc = Activation(self.encoder_activation[index])(enc)

        self._shape_before_dense = K.int_shape(enc)[1:]  # Ignore the batch size and take only the width, height and number of channels #

        # Dense NN #
        if (self.deep_dense):
            enc = Flatten()(enc)
            for index in range(self._num_encoder_layers):  #
                enc = Dense(units=self.encoder_neurons[index],
                            kernel_initializer=self.encoder_wt_initialization[index],
                            use_bias=False,
                            name=f"DENSE_ENCODER_LAYER_{index + 1}")(enc)

                enc = BatchNormalization(name=f"DENSE_BATCH_NORMALIZATION_FOR_ENCODER_LAYER_{index + 1}")(enc)
                enc = Activation(self.encoder_activation[index])(enc)

        else:
            enc = Flatten()(enc)
        return enc

    # Latent layer with Normal sampling #
    def _latent_layer(self, enc_lyr):
        self.mu=Dense(self.latent_neurons,
                      name="MEAN_VECTOR")(enc_lyr) # Mean vector #
        self.lv=Dense(self.latent_neurons,
                      name="LOG_VARIANCE_VECTOR")(enc_lyr) # LOG OF THE VARIANCE VECTOR #

        def sampling(inputs): # Method to sample a pont form a standard normal distribution #
            mu, lv = inputs
            ep=K.random_normal(shape=K.shape(self.mu))
            z = mu + (K.exp(lv/2.0)*ep)
            return z

        lec = Lambda(sampling,
                     name="LATENT_LAYER_WITH_NORMAL_SAMPLING")([self.mu, self.lv])
        lec = BatchNormalization(name=f"BATCH_NORMALIZATION_FOR_LATENT_LAYER")(lec)
        lec = Activation(self.latent_activation)(lec)
        return lec

    # --------------------------------DECODER--------------------------------#

    def _decoder_model(self):
        dec_inp = self._decoder_input()  # Input layer of the decoder, with neurons equal to the number of latent neurons #
        dec_lyr = self._decoder_layers(dec_inp)  # Stacking of the hidden layers of the decoder, excluding the output layer #
        dec_out = self._output_layer(dec_lyr)  # Output layer #
        self.decoder = Model(inputs=dec_inp,
                             outputs=dec_out,
                             name="DECODER")  # Decoder #

    # Decoder input layer #
    def _decoder_input(self):
        dec_inp = Input(shape=self.latent_neurons,
                        name="DECODER_INPUT")
        return dec_inp

    # Stacking the hidden layers of the decoder (DENSE + CONVOLUTIONAL) #
    def _decoder_layers(self, dec_inp):

        # Dense NN #
        if (self.deep_dense):
            dec = dec_inp
            dec = BatchNormalization(name="BATCH_NORMALIZATION_FOR_THE_INPUT_LAYER_OF_THE_DECODER")(dec)
            for index in range(self._num_decoder_layers):
                dec = Dense(units=self.decoder_neurons[index],
                            kernel_initializer=self.decoder_wt_initialization[index],
                            use_bias=False,
                            name=f"DECODER_LAYER_{index + 1}")(dec)

                dec = BatchNormalization(name=f"BATCH_NORMALIZATION_FOR_DECODER_LAYER_{index + 1}")(dec)
                dec = Activation(self.decoder_activation[index])(dec)
        else:
            dec = dec_inp
            dec = BatchNormalization(name="BATCH_NORMALIZATION_FOR_THE_INPUT_LAYER_OF_THE_DECODER")(dec)

        # Flattened shape of the convolutional before the dense/latent layer #
        dec = Dense(units=np.prod(self._shape_before_dense),
                    name="DENSE_DECODER_BEFORE_CONVOLUTION")(dec)

        # Reshape the dense layer to the shape of the last convolutional layer#
        dec = Reshape(self._shape_before_dense)(dec)

        # Convolutional NN #
        for index in reversed(range(1, self._num_conv_layers)):  # Convolutional NN
            dec = Conv2DTranspose(filters=self.conv_filters[index],
                                  kernel_size=self.conv_kernels[index],
                                  strides=self.conv_strides[index],
                                  padding=self.conv_padding[index],
                                  kernel_initializer=self.conv_wt_initialization[index],
                                  use_bias=False,
                                  name=f"CONVOLUTIONAL_DECODER_LAYER_{index + 1}")(dec)

            if (self.maxpooling):
                dec = MaxPooling2D(pool_size=self.pool_size[index],
                                    strides=self.pool_strides[index],
                                    padding=self.pool_padding[index])(dec)

            dec = BatchNormalization(name=f"CONVOLUTIONAL_BATCH_NORMALIZATION_FOR_DECODER_LAYER_{index + 1}")(dec)
            dec = Activation(self.encoder_activation[index])(dec)
        return dec


    # Output layer #
    def _output_layer(self, dec_lyr):
        dec_out = Conv2DTranspose(filters=1,
                                  kernel_size=self.conv_kernels[0],
                                  strides=self.conv_strides[0],
                                  padding=self.conv_padding[0],
                                  kernel_initializer='glorot_uniform',
                                  use_bias=True,
                                  name=f"DECODER_OUTPUT")(dec_lyr)
        dec_out = Activation('sigmoid')(dec_out)
        return dec_out


    # --------------------------------AUTOENCODER--------------------------------#
    def _autoencoder_model(self):
        input_model = self.encoder_input
        output_model = self.decoder(self.encoder(input_model))
        self.autoencoder = Model(inputs=input_model,
                                 outputs=output_model,
                                 name="AUTOENCODER")


    # --------------------------------SUMMARY OF THE AUTOENCODER--------------------------------#
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.autoencoder.summary()

    # --------------------------------VARIATIONAL_CONVOLUTIONAL_AUTOENCODER_LOSS--------------------------------#
    def _total_loss(self, target_value, predicted_value):
        rec_loss=_reconstruction_loss(target_value, predicted_value)
        kl_loss=_kl_loss(self)()
        total_loss=(self.rec_loss_wt * rec_loss) + (self.beta * kl_loss)
        return total_loss

    # --------------------------------COMPILING THE AUTOENCODER--------------------------------#
    def compile(self,
                optimizer=Adam,
                learning_rate=0.001):
        self.autoencoder.compile(optimizer=optimizer(learning_rate),
                                 loss=self._total_loss,
                                 metrics=[_reconstruction_loss, _kl_loss(self)])

    # --------------------------------TRAINING THE AUTOENCODER--------------------------------#
    def train(self,
              x_train=None,
              x_test=None,
              y_train=None,
              y_test=None,
              batch_size=100,
              epochs=100,
              shuffle=True):
        self.autoencoder.fit(x=x_train,
                             y=y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             shuffle=shuffle,
                             validation_data=(x_test, y_test))  # x_train=training data, x_test=testing data, y_train=labels of the training data, y_test=labels of the testing data #


    # --------------------------------SAVING THE AUTOENCODER--------------------------------#
    def save(self, save_dir_name="."):
        # Create the directory to save the autoencoder model #
        check_dir = os.path.isdir(save_dir_name)  # Checks if the specified directory exists #
        if not check_dir:  # If directory does not exist, then create it #
            os.makedirs(save_dir_name, exist_ok=False)
            print("Created directory : ", save_dir_name)
        else:
            os.makedirs(save_dir_name, exist_ok=True)  # If directory exists, then overwrite it #
            print("Overwritten directory : ", save_dir_name)

        # Saving the parameters, weights and loss of the autoencoder model #
        parameters = [self.input_shape,
                      self.encoder_neurons,
                      self.encoder_activation,
                      self.encoder_wt_initialization,
                      self.latent_neurons,
                      self.latent_activation,
                      self.latent_wt_initialization,
                      self.decoder_neurons,
                      self.decoder_activation,
                      self.decoder_wt_initialization,
                      self.conv_filters,
                      self.conv_kernels,
                      self.conv_strides,
                      self.conv_padding,
                      self.conv_activation,
                      self.conv_wt_initialization,
                      self.pool_size,
                      self.pool_strides,
                      self.pool_padding,
                      self.rec_loss_wt,
                      self.beta,
                      self.deep_dense,
                      self.maxpooling]

        parameters_filename = os.path.join(save_dir_name, "parameters.pkl")  # Path and name of the file where the parameters would be saved #
        with open(parameters_filename, "wb") as file:
            pickle.dump(parameters, file)

        # Saving the weights of the autoencoder #
        weights_filename = os.path.join(save_dir_name, "weights.h5")  # Path and name of the file where the weights would be saved #
        self.autoencoder.save_weights(weights_filename)

        # Saving the training loss and validation loss #
        training_loss_filename = os.path.join(save_dir_name, "training_loss.dat")  # Path and name of the file where the training loss would be saved #
        np.savetxt(training_loss_filename, self.autoencoder.history.history['loss'])  # Training loss #

        validation_loss_filename = os.path.join(save_dir_name, "validation_loss.dat")  # Path and name of the file where the validation loss would be saved #
        np.savetxt(validation_loss_filename, self.autoencoder.history.history['val_loss'])  # Validation loss #


    # --------------------------------LOADING THE AUTOENCODER--------------------------------#
    def _load_weights(self, weights_filename):
        self.autoencoder.load_weights(weights_filename)


    @classmethod
    def load(cls, save_dir_name="."):
        # Load the parameters #
        parameters_filename = os.path.join(save_dir_name, "parameters.pkl")
        with open(parameters_filename, "rb") as file:
            parameters = pickle.load(file)
        autoencoder = variational_convolutional_autoencoder(*parameters)

        # Load the weights #
        weights_filename = os.path.join(save_dir_name, "weights.h5")
        autoencoder._load_weights(weights_filename)
        return autoencoder

    # --------------------------------RECONSTRUCTION--------------------------------#
    def reconstruct_latent(self, data):
        latent_data = self.encoder.predict(data)  # Get the latent data #
        return latent_data


    def reconstruct_input(self, data):
        latent_data = self.encoder.predict(data)  # Get the latent data #
        reconstructed_data = self.decoder.predict(latent_data)  # Get the reconstructed data #
        return reconstructed_data

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
