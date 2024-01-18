#!/home/subinoy/Softwares/anaconda3/envs/tf/bin/python3

# OWNER : SUBINOY ADHIKARI           
# EMAIL : subinoyadhikari@tifrh.res.in

import os, sys, stat
import gc
import pickle
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from dense_variational_autoencoder import dense_variational_autoencoder
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import numpy as np
import random


################################################################################################################################################################################################################################################

SEED = 0

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=SEED)

# Call the above function with seed value
set_global_determinism(seed=SEED)


# Run the script in GPU
gpu_id="0"
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id # Model will be trained on the specified GPU #

################################################################################################################################################################################################################################################

# Method to split the data into a training set and testing set
def split_train_test(ca_contact_scaled):
    input_cv = ca_contact_scaled
    count = 50

    #-----Determine the percentage of the entire dataset to be used for training-----#
    percentage_of_training_data = 80

    #-----Determine the percentage of the entire dataset to be used for testing-----#
    percentage_of_testing_data = 100 - (percentage_of_training_data)

    #-----Determining the size of the training data-----#
    no_of_training_data = int(count*int(0.01*percentage_of_training_data*input_cv.shape[0]/count))

    #-----Determining the size of the testing data-----#
    no_of_testing_data = input_cv.shape[0] - no_of_training_data

    #-----Determine the training indexes-----#
    training_indexes = np.linspace(0, input_cv.shape[0]-1, no_of_training_data).astype('int')

    #-----Determine the testing indexes-----#
    testing_indexes = np.array(list(set(np.arange(input_cv.shape[0]).tolist()).difference(training_indexes.tolist())))
    #-----Generate the training data-----#
    training_data = []

    for i in range(len(training_indexes)):
        training_data.append(input_cv[training_indexes[i]])

    training_data = np.array(training_data)

    #-----Generate the testing data-----#
    testing_data = []

    for i in range(len(testing_indexes)):
        testing_data.append(input_cv[testing_indexes[i]])

    testing_data = np.array(testing_data)

    #-----Verify the complete, training and testing dataset-----#

#     print("Length of total dataset = ", input_cv.shape[0])
#     print("Length of training dataset = ", len(training_data))
#     print("Length of testing dataset = ", len(testing_data))
#     print("Length of (training + testing) dataset = ", len(training_data)+len(testing_data))
    
    
    return training_data, testing_data

################################################################################################################################################################################################################################################


if __name__ == "__main__":

	# Starting point
	start=0

	# Number of dimensions
	input_shape = 9453 # Check this value before manually putting it

	# Path to the scaled Ca pairwise distance data
	path="/path/to/data/"

	# Load the scaled pairwise Ca distance data
	ca_contact_scaled = np.load(path+f"data.npy")

	# Training and testing data
	x_train, x_test = split_train_test(ca_contact_scaled=ca_contact_scaled)     
	    
	# Parameters to run the autoencoder
	# -----ENCODER-----#
	encoder_neurons=[4096, 512, 128,  16]
	encoder_activation=['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh']
	encoder_wt_initialization=['glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

	# -----LATENT-----#
	latent_neurons=2
	latent_activation='tanh'
	latent_wt_initialization='glorot_uniform'

	# -----DECODER-----#
	decoder_neurons=[16, 128, 512, 4096]
	decoder_activation=['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh']
	decoder_wt_initialization=['glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

	optimizer=Adam
	loss=MeanSquaredError()
	learning_rate=1e-4
	batch_size=64
	epochs=300
	shuffle=True     
	alpha=10
	beta=1e-12   

	# -----SAVING FILENAME-----#
	save_dir_init=f"start_{start}_asyn_fasudil_latent_{latent_neurons}"
	save_dir_name=f"{save_dir_init}_batch_{batch_size}_epochs_{epochs}"

	# Build and compile the model
	ae = dense_variational_autoencoder(input_shape=input_shape,
				           encoder_neurons=encoder_neurons,
				           encoder_activation=encoder_activation,
				           encoder_wt_initialization=encoder_wt_initialization,
				           latent_neurons=latent_neurons,
				           latent_activation=latent_activation,
				           latent_wt_initialization=latent_wt_initialization,
				           decoder_neurons=decoder_neurons,
				           decoder_activation=decoder_activation,
				           decoder_wt_initialization=decoder_wt_initialization,
				           alpha=alpha,
				           beta=beta)
	ae.summary()
	ae.compile(optimizer=optimizer,
		       learning_rate=learning_rate)         
		
	# Train the autoencoder
	ae.train(x_train=x_train,
		     x_test=x_test,
		     y_train=x_train,
		     y_test=x_test,
		     batch_size=batch_size,
		     epochs=epochs,
		     shuffle=shuffle)
	     
	ae.save(save_dir_name=save_dir_name) 	     
	print(f"saved directory = {save_dir_name}")        
 

################################################################################################################################################################################################################################################

# Free the memory and quit
for element in dir():
    if element[0:2] != "__":
        del element
gc.collect()
sys.exit(0)         
