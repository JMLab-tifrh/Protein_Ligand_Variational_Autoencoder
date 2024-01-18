#!/home/jm/Softwares/anaconda3/envs/tf/bin/python3

# OWNER : SUBINOY ADHIKARI           
# EMAIL : subinoyadhikari@tifrh.res.in

import os, sys, stat
import gc
import pickle
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from denoising_variational_convolutional_autoencoder import variational_convolutional_autoencoder
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import random


################################################################################################################################################################################################################################################

SEED = 43

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
    count = 10

    #-----Determine the percentage of the entire dataset to be used for training-----#
    percentage_of_training_data = 90

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


# Path to the scaled Ca pairwise distance data for asyn-lig47
path="/path/to/contact_map/"

noise_factor=0.035
scaler=MinMaxScaler()

#################### ASYN FASUDIL ####################

ca_contact_scaled_asyn_fas = np.load(path+f"data_asyn_fas.npy") # Load the #scaled pairwise Ca distance data
ca_contact_scaled_asyn_fas = np.array([np.pad(ca_contact_scaled_asyn_fas[i], pad_width=2) for i in range(len(ca_contact_scaled_asyn_fas))]) # Pad the data so that it becomes 144x144

# Training and testing data
x_train_asyn_fas, x_test_asyn_fas = split_train_test(ca_contact_scaled=ca_contact_scaled_asyn_fas)    

# Creating the noisy dataset

noisy_xtrain_asyn_fas=[]
noisy_xtest_asyn_fas=[]

for cmap in x_train_asyn_fas:
	noisy_cmap=cmap + noise_factor*np.random.random(cmap.shape)
	#noisy_cmap = np.clip(noisy_cmap, 0, 1)
	noisy_xtrain_asyn_fas.append(scaler.fit_transform(noisy_cmap))
noisy_xtrain_asyn_fas=np.array(noisy_xtrain_asyn_fas)


for cmap in x_test_asyn_fas:
	noisy_cmap=cmap + noise_factor*np.random.random(cmap.shape)
	#noisy_cmap = np.clip(noisy_cmap, 0, 1)
	noisy_xtest_asyn_fas.append(scaler.fit_transform(noisy_cmap))
noisy_xtest_asyn_fas=np.array(noisy_xtest_asyn_fas)


x_train_asyn_fas = x_train_asyn_fas.reshape(x_train_asyn_fas.shape+(1,))   
noisy_xtrain_asyn_fas = noisy_xtrain_asyn_fas.reshape(noisy_xtrain_asyn_fas.shape+(1,))

x_test_asyn_fas = x_test_asyn_fas.reshape(x_test_asyn_fas.shape+(1,))
noisy_xtest_asyn_fas = noisy_xtest_asyn_fas.reshape(noisy_xtest_asyn_fas.shape+(1,))

#################### ASYN ####################

ca_contact_scaled_asyn = np.load(path+f"data_asyn.npy") # Load the #scaled pairwise Ca distance data
ca_contact_scaled_asyn = np.array([np.pad(ca_contact_scaled_asyn[i], pad_width=2) for i in range(len(ca_contact_scaled_asyn))]) # Pad the data so that it becomes 144x144

# Training and testing data
x_train_asyn, x_test_asyn = split_train_test(ca_contact_scaled=ca_contact_scaled_asyn) 

# Creating the noisy dataset

noisy_xtrain_asyn=[]
noisy_xtest_asyn=[]

for cmap in x_train_asyn:
	noisy_cmap=cmap + noise_factor*np.random.random(cmap.shape)
	noisy_cmap = np.clip(noisy_cmap, 0, 1)
	noisy_xtrain_asyn.append(scaler.fit_transform(noisy_cmap))
noisy_xtrain_asyn=np.array(noisy_xtrain_asyn)


for cmap in x_test_asyn:
	noisy_cmap=cmap + noise_factor*np.random.random(cmap.shape)
	noisy_cmap = np.clip(noisy_cmap, 0, 1)
	noisy_xtest_asyn.append(scaler.fit_transform(noisy_cmap))
noisy_xtest_asyn=np.array(noisy_xtest_asyn)


x_train_asyn = x_train_asyn.reshape(x_train_asyn.shape+(1,))   
noisy_xtrain_asyn = noisy_xtrain_asyn.reshape(noisy_xtrain_asyn.shape+(1,))

x_test_asyn = x_test_asyn.reshape(x_test_asyn.shape+(1,))
noisy_xtest_asyn = noisy_xtest_asyn.reshape(noisy_xtest_asyn.shape+(1,))

####################################################################################################

noisy_xtrain_all=np.vstack([noisy_xtrain_asyn_fas, noisy_xtrain_asyn])
noisy_xtest_all=np.vstack([noisy_xtest_asyn_fas, noisy_xtest_asyn])

x_train_all=np.vstack([x_train_asyn_fas, x_train_asyn])
x_test_all=np.vstack([x_test_asyn_fas, x_test_asyn])

####################################################################################################
  
# -----PARAMETERS-----#
optimizer = Adam
loss = MeanSquaredError()
learning_rate = 0.0005
batch_size = 32
epochs = 300
shuffle = True

save_dir_init = f"cnn_vae_asyn"
save_dir_name = f"{save_dir_init}_batch_{batch_size}_epochs_{epochs}"

# Build and compile the model
ae=variational_convolutional_autoencoder(	input_shape=(144,144,1),
						dense_encoder_neurons=[4096, 2048,1024, 256, 64, 16],
						dense_encoder_activation=['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh'],
						dense_encoder_wt_initialization=['glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform',
						'glorot_uniform', 'glorot_uniform'],
						dense_latent_neurons=2,
						dense_latent_activation='tanh',
						dense_latent_wt_initialization='glorot_uniform',
						dense_decoder_neurons=[16, 64, 256, 1024, 2048, 4096, ],
						dense_decoder_activation=['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh'],
						dense_decoder_wt_initialization=['glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform',
						'glorot_uniform', 'glorot_uniform'],
						conv_filters=(16,32,64,96,128),
						conv_kernels=(3,3,3,3,3),
						conv_strides=(1,2,2,2,2,1),
						conv_padding=['same', 'same', 'same', 'same', 'same', 'same', 'same', 'same', 'same', 'same', 'same', 'same', 'same',  ],
						conv_activation=['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh'],
						conv_wt_initialization=['glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', 										'glorot_uniform'],
						pool_size=[2,2],
						pool_strides=[1,1],
						pool_padding=['same','same', 'same', 'same', 'same', 'same', 'same', 'same', 'same', 'same', 'same', 'same', 'same',  ],
						rec_loss_wt=10.0,
						beta=1.0e-12,
						deep_dense=True,
						maxpooling=False)
						

ae.summary()
ae.compile(optimizer=optimizer,
           learning_rate=learning_rate)

print(x_train_all.shape)
# Train the autoencoder
ae.train(x_train=noisy_xtrain_all,
         x_test=noisy_xtest_all,
         y_train=x_train_all,
         y_test=x_test_all,
         batch_size=batch_size,
         epochs=epochs,
         shuffle=shuffle)

ae.save(save_dir_name=save_dir_name)
	     
print(f"saved directory = {save_dir_name}")    
del ca_contact_scaled_asyn_fas
del ca_contact_scaled_asyn
del x_train_all
del x_test_all
del ae
gc.collect()         
 

################################################################################################################################################################################################################################################

# Free the memory and quit
for element in dir():
    if element[0:2] != "__":
        del element
gc.collect()
sys.exit(0)         
