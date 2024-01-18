This is the repository for variational autoencoder (VAE) and denoising convolutional VAE (DCVAE) written in tensorflow. Instructions to run the codes are in the "README.dat" file of the individual directories

#----------REQUIRED PACKAGES----------#

1. TensorFlow
2. NumPy 
3. Scikit-learn

#----------PROGRAM INPUTS----------#

For both VAE and DCVAE models, one requires a file that contains the training and testing data in *.npy format. 

#----------PROGRAM OUTPUTS----------#

A directory containing the training and validation loss and the model parameters.

#----------HOW TO RUN THE VAE PROGRAM----------#

1. Modify the following variables for a different protein in the file "train_dvae.py" :
   
	a. input_shape
	b. path
	c. filename as in "ca_contact_scaled = np.load(path+f"data.npy")". Change "data.npy" to "your_filename.npy".
	d. Change the number of encoder, latent and decoder neurons as per requirement.
	e. Change the optimizer, loss, learning_rate, batch_size, epochs, alpha and beta as per requirement.
  
  	2. Type "./train_dvae.py" in the terminal with the environment activated with the specified packages and hit Enter.
  
  	3. To create the latent data and reconstruct the input open a jupyter notebook and type the following statements   	       
		  ate=dense_variational_autoencoder.load(save_dir_name) # Name of the directory where the model parameters are saved. 
		  ca_contact_scaled = np.load(path+f"data.npy") # Load the data
		  latent_data=ate.reconstruct_latent(ca_contact_scaled) # Construct the latent data
		  reconstructed_data = ate.reconstruct_input(ca_contact_scaled) # Reconstruct the input data

   #----------HOW TO RUN THE DCVAE PROGRAM----------#

   1. Modify the following variables for a different protein in the file "train_denoising_vae.py" :
      
	a. path
	b. filename as in "ca_contact_scaled_asyn_fas = np.load(path+f"data_asyn_fas.npy")". Change "data_asyn_fas.npy" to "your_filename.npy". Reshape the data to height x width x depth. 
	c. Change the input_shape, filters, kernels, strides, padding, number of encoder, latent and decoder neurons as per requirement.
	d. Change the optimizer, loss, learning_rate, batch_size, epochs and beta as per requirement.
3. Type "./train_denoising_vae.py" in the terminal with the environment activated with the specified packages and hit Enter.
4. To create the latent data and reconstruct the input :
	        # Open a jupyter notebook and load the model in a cell as 
		ate=variational_convolutional_autoencoder.load(save_dir_name) # Name of the directory where the model parameters are saved. 
		ca_contact_scaled = np.load(path+f"data_asyn_fas.npy") # Load the data
		latent_data = ate.reconstruct_latent(ca_contact_scaled)
		reconstructed_data = ate.reconstruct_input(ca_contact_scaled)

      

