import pandas as pd
import numpy as np
import os
import scipy.io
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, LSTM, Attention, Concatenate, Lambda
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import normalize
from scipy import stats

"""
Things to do
- Evaluate the efficacy of the AE (adjust the hyperparameters)
- Figure out how to get directionlity of connections
	- Could take the correlatin of the 2nd AE node time series with the phenotype
- Check the predictability of the model for benchmarking
- Evaluate the amount of loss for both of the autoencoders
- Change the softmax function to be a one-sided test excluding values over the median
"""

folder_path = '/home/nbrady/Desktop/deep_functional_net'
phenotypes = pd.read_csv(f'{folder_path}/PHEN_MATRIX.csv')

# Exclude these Subs for FD issues
ignore_subs = [5027, 5011, 5140, 5142, 5172, 5036, 5106]
included_subs = phenotypes['sid_rise']

# Phenotype of interest
phen_var = 'STATE_Tot_all_pn1'
phen_data = np.array([])

# Define a similarity metric (e.g., cosine similarity)
def cosine_similarity(a, b):
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)

def softmax(matrix, axis=1):
	mean_value = np.mean(matrix)
	# Create a mask to exclude the values over mean
	binary_mask = (matrix > mean_value).astype(int)

	# Actual softmax function
	exponentiated_values = np.exp(-(matrix - mean_value))
	sum_exponentiated_values = np.sum(exponentiated_values, axis=axis, keepdims=True)
	softmax_result = exponentiated_values / sum_exponentiated_values

	# Mask values over mean
	masked_result = softmax_result + binary_mask	
	return masked_result

def dynamic_time_warp(x, y):
	# Create the distance matrix.
	D = np.zeros((len(x), len(y)))

	for i in range(len(x)):
		for j in range(len(y)):
			D[i, j] = np.abs(x[i] - y[j])
			
	# Compute the cumulative distance matrix.
	for i in range(1, len(x)):
		for j in range(1, len(y)):
			D[i, j] += np.min([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
													    
	# Return the distance at the last row and column of the matrix.
	return D[-1, -1]


#=======================================================
# Load in the bold timeseries to create Nx268x268 matrix
#=======================================================
num_subs = len(os.listdir(f"{folder_path}/timeseries"))
timeseries_compression = 100

all_data = np.zeros((num_subs, 268, timeseries_compression))
#all_data = np.zeros((5, 268, timeseries_compression))

ts_data = os.listdir(f"{folder_path}/timeseries")

for sub_id, sub in enumerate(ts_data):

	# Ignore subjects with missing data or with high FD values
	rise_id = int(sub.split('_')[0].split('-')[-1].split('m')[0])
	sub_df = phenotypes[phenotypes['sid_rise'] == rise_id]
	sub_phen_val = sub_df[phen_var].values[0] if len(sub_df[phen_var].values) > 0 else None
	use_data = (rise_id not in ignore_subs) or (sub_phen_val != None)

	if use_data == True:

		phen_data = np.append(phen_data, sub_phen_val)
		path = f"{folder_path}/timeseries/{sub}"
		
		# Read in the CSV
		data = pd.read_csv(path)
		data = data.T
		data = data.drop('Unnamed: 0')

		#print("Shape of input data: ", data.shape)

		#=================================================================
		# Train the autoencoder for y dimension reduction for each subject
		#=================================================================

		np.random.seed(0)

		# Autoencoder architecture
		input_dim = data.shape[1]
		latent_dim = timeseries_compression  # Adjust based on your requirements

		encoder = models.Sequential([
			    layers.InputLayer(input_shape=(None, input_dim)),
			    layers.LSTM(latent_dim, activation='relu', return_sequences=False),
			    ])

		decoder = models.Sequential([
			layers.RepeatVector(latent_dim),
			layers.LSTM(input_dim, activation='sigmoid', return_sequences=True)
			])

		autoencoder = models.Sequential([encoder, decoder])

		autoencoder.compile(optimizer='adam', loss='mse')

		# Train the autoencoder
		autoencoder.fit(data, data, epochs=100, batch_size=16, shuffle=True)

		# Encode the data to obtain the latent representations
		latent_rep= encoder.predict(data)
		#print("latent_rep shape: ", latent_rep.shape)

		
		# Generate reduced dimensionality data
		reconstructed_timeseries = decoder.predict(latent_rep)

		# Test accuracy of the compression
		result = autoencoder.evaluate(data, reconstructed_timeseries)
		#print(result)

		# Store the reduced data into all_data
		all_data[sub_id] = latent_rep

		# Uncomment for testing
		#if sub_id > 3:
			#break

# Keep only the arrays with data populated from subjects (rest are zero placeholders)
all_data = all_data[:phen_data.size,:,:]
print("All Data Shape: ", all_data.shape, "Phenotype data: ", phen_data.shape)
flat_df = pd.DataFrame(all_data.flatten())
flat_df.to_csv(f"{folder_path}/encoded_timeseries.csv", index=False, header=False)

#========================================================
# Incorporate the Phenotype as a weight to the timeseries
# Also find the correlation for directionality
#========================================================

r_val = np.zeros((num_subs, 268, 268))

for sub_pos in range(len(all_data)):
	phen_val = phen_data[sub_pos]
	all_data[sub_pos,:,:] = all_data[sub_pos,:,:] * phen_val

#============================================ 
# Train autoencoder for z dimension reduction 
#============================================

data = np.reshape(all_data, (all_data.shape[0], -1)).T

print("Data Shape before 2nd AE: ", data.shape)

input_shape = (data.shape[1],)
encoding_dim = 100
bottle_neck_dim = 50

z_encoder = models.Sequential([
	layers.Dense(encoding_dim, activation='relu', input_shape=input_shape),
	layers.Dense(bottle_neck_dim, activation='relu'),
	layers.Dense(1, activation='sigmoid')
	])

z_decoder = models.Sequential([
	layers.InputLayer(input_shape=(1,)),
	layers.Dense(1, activation='relu'),
	layers.Dense(bottle_neck_dim, activation='relu'),
	layers.Dense(encoding_dim, activation='sigmoid'),
	])

# Compile the model
z_encoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
z_encoder.fit(data, data, epochs=100, batch_size=16, shuffle=True)

# Use the trained autoencoder to encode the data
encoded_data = z_encoder.predict(data)
print("Encoded Data Shape: ", encoded_data.shape)

# Reshape the encoded data back to the desired shape
#final_encoded_data = np.reshape(encoded_data, (1, data.shape[0], data.shape[1]))
final_encoded_data = np.reshape(encoded_data, (268, 100))
data = final_encoded_data

print("Final encoded data: ", final_encoded_data, final_encoded_data.shape)

#=================================== 
# Calculate node level similarities 
#===================================

# Compute pairwise similarities
pairwise_similarities = np.zeros((len(data), len(data)))
print("Length of data: ", len(data))

for i in range(len(data)):
	for j in range(len(data)):
		pairwise_similarities[i, j] = cosine_similarity(latent_rep[i], latent_rep[j])
		#pairwise_similarities[i, j] = dynamic_time_warp(latent_rep[i], latent_rep[j])

"""
# Uncomment this for DTW
# Normalize the dynamic time warping results to a p-value
pairwise_similarities = softmax(pairwise_similarities, axis=1)
print("Pairwise Similarities")
print(pairwise_similarities)
pw_df = pd.DataFrame(pairwise_similarities)
pw_df.to_csv('./pairwise_similarities.csv')
"""
# Set a threshold for clustering
threshold = 0.90 # for cosine similarity
#threshold = 0.01 # for DTW with softmax

# Apply threshold for clustering
clusters = []
for i in range(len(data)):
	cluster = [j for j in range(len(data)) if pairwise_similarities[i, j] > threshold]
	clusters.append(cluster)

print("Clusters:", clusters)


#========================================
# Create a sparse matrix for connectivity
#========================================
	
fc_matrix = np.zeros((268, 268))

print("Generating the connectivity matrix")
for fc in clusters:
	if len(fc) > 1:
		#print(fc)
		for target_id, target_node in enumerate(fc):
			for id, node in enumerate(fc):
				if id != target_id:
					#print("Connection made: ", target_node, node)
					fc_matrix[target_node, node] = 1



#==================================================
# Train another model to incorporate the phenotypes
#==================================================

print("Saving the matrix as csv")
df = pd.DataFrame(fc_matrix)
print(df.shape)
df.to_csv(f'{folder_path}/testing_autoencoder_matrix.csv', index=False, header=False)
