import pandas as pd
import numpy as np
import os
import scipy.io
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Attention, Concatenate, Lambda, Masking
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import normalize
from scipy import stats
import matplotlib.pyplot as plt

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
    
    
#=======================================================
# Load in the bold timeseries to create Nx268x268 matrix
#=======================================================
num_subs = len(os.listdir(f"{folder_path}/timeseries"))
timeseries_compression = 100

#all_data = np.zeros((num_subs, 2736, 268))
#all_data = np.zeros((5, 268, timeseries_compression))
all_data = []

ts_data = os.listdir(f"{folder_path}/timeseries")

for sub_id, sub in enumerate(ts_data):
    # Ignore subjects with missing data or with high FD values
    rise_id = int(sub.split('_')[0].split('-')[-1].split('m')[0])
    sub_df = phenotypes[phenotypes['sid_rise'] == rise_id]
    sub_phen_val = sub_df[phen_var].values[0] if len(sub_df[phen_var].values) > 0 else None
    use_data = (rise_id not in ignore_subs) or (sub_phen_val != None)
    
    if use_data == True:
        path = f"{folder_path}/timeseries/{sub}"
        # Read in the CSV
        data = pd.read_csv(path)
        data = data.T
        data = data.drop('Unnamed: 0')
        data = data.T
        all_data.append(data)
        #print(data.shape, sub_id, sub)
        
# Pad sequences
padded_data = pad_sequences(all_data, padding='post')

#=================================================================
# Train the autoencoder for y dimension reduction for each subject
#=================================================================

# Autoencoder architecture
input_shape = (padded_data.shape[1], padded_data.shape[2])
print(padded_data.shape[0], padded_data.shape[1], padded_data.shape[2])
latent_dim = 100  # Adjust based on your requirements

# Encoder with additional hidden layers
encoder = models.Sequential([
    layers.InputLayer(input_shape=input_shape),
    layers.Masking(mask_value=0.),  # Masking layer to ignore zeros
    layers.LSTM(1024, activation='tanh', return_sequences=True),  # More neurons and return sequences
    layers.LSTM(512, activation='tanh', return_sequences=True),  # More neurons and return sequences
    layers.LSTM(256, activation='tanh', return_sequences=True),  # Gradual compression
    layers.LSTM(latent_dim, activation='tanh', return_sequences=False),  # Bottleneck layer
])

# Decoder with additional hidden layers, mirroring the encoder's structure
decoder = models.Sequential([
    layers.RepeatVector(padded_data.shape[1]),  # all_data.shape[1] should be the number of timesteps
    layers.LSTM(256, activation='tanh', return_sequences=True),  # Symmetrical expansion
    layers.LSTM(512, activation='tanh', return_sequences=True),  # More neurons
    layers.LSTM(1024, activation='tanh', return_sequences=True),  # More neurons
    layers.LSTM(padded_data.shape[2], activation='sigmoid', return_sequences=True)  # Output layer to match input features
])

autoencoder = models.Sequential([encoder, decoder])

autoencoder.compile(optimizer='Adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(padded_data, padded_data, epochs=100, validation_split=0.2, batch_size=16, shuffle=True,)

# Generate synthetic data (example using random noise as input)
noise = np.random.normal(size=(50, latent_dim))
synthetic_data = decoder.predict(noise)

# Plotting
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
plt.savefig('autoencoder_loss.png')


#========================================
# Evaluate the models performance
#========================================
reconstructed_data = autoencoder.predict(all_data)
mse = np.mean(np.square(all_data - reconstructed_data))
cossim = cosine_similarity(all_data, reconstructed_data)
print(f"Mean Squared Error: {mse}, Cosine Similarity: {cossim}")

#========================================
# Download the model
#========================================
# Save the entire model as a `.keras` zip archive.
decoder.save('cpm_synthetic_data_model.keras')
