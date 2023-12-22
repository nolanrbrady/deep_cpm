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
