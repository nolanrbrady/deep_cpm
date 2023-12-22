import pandas as pd
import numpy as np
import os
import scipy.io
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import normalize
from scipy import stats


"""
data = pd.read_csv("./timeseries/sub-5062mom_realcry_denoised_bold_timeseries.csv")
data = data.T
data = data.drop('Unnamed: 0')
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

#all_data = np.zeros((num_subs, 268, timeseries_compression))
all_data = np.zeros((5, 268, timeseries_compression))

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
			    layers.InputLayer(input_shape=(input_dim,)),
			    layers.Dense(latent_dim, activation='relu'),
			    ])

		decoder = models.Sequential([
			layers.InputLayer(input_shape=(latent_dim,)),
			layers.Dense(input_dim, activation='sigmoid')
			])

		autoencoder = models.Sequential([encoder, decoder])

		autoencoder.compile(optimizer='adam', loss='mse')

		# Train the autoencoder
		autoencoder.fit(data, data, epochs=10, batch_size=16, shuffle=True)

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
		if sub_id > 3:
			break

print("All Data Shape: ", all_data.shape, "Phenotype data: ", phen_data.shape)
flat_df = pd.DataFrame(all_data.flatten())
flat_df.to_csv(f"{folder_path}/encoded_timeseries.csv", index=False, header=False)

#============================================ 
# Train autoencoder for z dimension reduction 
#============================================

data = np.reshape(all_data, (data.shape[0], -1))

input_shape = (data.shape[1],)
encoding_dim = 100
bottle_neck_dim = 50

z_encoder = models.Sequential([
	layers.Dense(encoding_dim, activation='relu', input_shape=input_shape),
	layers.Dense(bottle_neck_dim, activation='relu'),
	layers.Dense(encoding_dim, activation='relu'),
	layers.Dense(data.shape[1], activation='sigmoid')
	])

# Compile the model
z_encoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
z_encoder.fit(data, data, epochs=10, batch_size=16, shuffle=True)

# Use the trained autoencoder to encode the data
encoded_data = z_encoder.predict(data)

# Reshape the encoded data back to the desired shape
final_encoded_data = np.reshape(encoded_data, (1, data.shape[0], data.shape[1]))

#=================================== 
# Calculate node level similarities 
#===================================

# Compute pairwise similarities
pairwise_similarities = np.zeros((len(data), len(data)))

for i in range(len(data)):
	for j in range(len(data)):
		#pairwise_similarities[i, j] = cosine_similarity(latent_rep[i], latent_rep[j])
		pairwise_similarities[i, j] = dynamic_time_warp(latent_rep[i], latent_rep[j])

# Normalize the dynamic time warping results using modified z-score for outliers
pw_median = np.median(pairwise_similarities)
pw_mad = np.median(np.abs(pairwise_similarities) - pw_median)

pairwise_similarities = 0.6745 * (pairwise_similarities - median) / mad
print("Pairwise simialarities: ", pairwise_similarities)

# Set a threshold for clustering
threshold = 1.645

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
