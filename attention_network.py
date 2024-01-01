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


#======================================================						
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
		 # Assuming your timeseries data is stored in a variable named 'timeseries_data'
		 # Shape of timeseries_data: (num_subjects, 268, 100)

		 # Assuming your behavioral scores are stored in a variable named 'behavioral_scores'
		 # Shape of behavioral_scores: (num_subjects,)

		 # Reshape the timeseries data to (num_subjects, 26800)
		 data = np.reshape(timeseries_data, (timeseries_data.shape[0], -1))

		 # Define the neural network model
		 input_timeseries = Input(shape=(data.shape[1],))

		 # Add attention mechanism to focus on important regions
		 attention_probs = Dense(data.shape[1], activation='softmax', name='attention_probs')(input_timeseries)
		 attention_mul = Concatenate()([input_timeseries, attention_probs])

		 # Add a dense layer for prediction
		 output_behavior = Dense(1, activation='linear')(attention_mul)

		 # Create the model
		 model = Model(inputs=input_timeseries, outputs=[output_behavior, attention_probs])

		 # Compile the model
		 model.compile(optimizer='adam', loss='mean_squared_error')

		 # Train the model
		 model.fit(data, [phen_data, np.zeros_like(phen_data)], epochs=50, batch_size=32, shuffle=True)

		 # Get the attention weights for a specific subject
		 subject_index = 0
		 attention_weights = model.predict(data[subject_index:subject_index+1])[1]
		 print("Attention weights: ", attention_weights, attention_weights.shape)
	break






