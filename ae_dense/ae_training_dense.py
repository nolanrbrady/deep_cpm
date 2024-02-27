import pandas as pd
import numpy as np
import os
import scipy.io
import re
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Attention, Concatenate, Lambda, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
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

#==============================================
# Split into training, validation and test set
#==============================================

train_data, test_data = train_test_split(padded_data, test_size=0.2, random_state=42)

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
print(train_data.shape[0], train_data.shape[1], train_data.shape[2])

#=================================================================
# Train the autoencoder for y dimension reduction for each subject
#=================================================================

early_stopping = EarlyStopping(
	monitor='val_loss',  
	patience=25,        
	verbose=1,         
	restore_best_weights=True)

checkpoint_filepath = './best_ae_model_weight.h5'  # Path to save the model file

model_checkpoint_callback = ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,  # Set to False to save the whole model
	monitor='val_loss',
	mode='min',
	save_best_only=True,  # Only save a model if `val_loss` has improved
	verbose=1)

# Autoencoder architecture
#input_shape = (padded_data.shape[1], padded_data.shape[2])
#print(padded_data.shape[0], padded_data.shape[1], padded_data.shape[2])

#input_shape = (train_data.shape[1], train_data[2])
input_shape = (36209, 268)
output_shape = input_shape[0] * input_shape[1]

print("Input Shape: ", input_shape)
print("Output Shape: ", output_shape)
print(train_data.shape[0], train_data.shape[1], train_data.shape[2])
# should give shape (65, 36209, 268)

latent_dim = 128  # Adjust based on your requirements

# Encoder with additional hidden layers
encoder = models.Sequential([ 
    layers.Masking(mask_value=0., input_shape=input_shape),  # Masking layer to ignore zeros
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),  # More neurons and return sequences
    layers.Dense(512, activation='relu'),  # More neurons and return sequences
    layers.Dense(256, activation='relu'),  # Gradual compression
    layers.Dense(latent_dim, activation='relu'),  # Bottleneck layer
])

# Decoder with additional hidden layers, mirroring the encoder's structure
decoder = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(latent_dim,)),  # Symmetrical expansion
    layers.Dense(512, activation='relu'),  # More neurons
    layers.Dense(1024, activation='relu'),  # More neurons
    layers.Dense(output_shape, activation='relu'),  # Output layer to match input features
    layers.Reshape((36209, 268))
])

autoencoder = models.Sequential([encoder, decoder])

autoencoder.compile(optimizer='Adam', loss='mse')

# Check if checkpoint exists
if os.path.exists(checkpoint_filepath):
	print("Checkpoint found. Loading weights.")
	autoencoder.load_weights(checkpoint_filepath)
else:
	print("No checkpoint found. Starting training from scratch.")


# Set up TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the autoencoder
history = autoencoder.fit(
		train_data, 
		train_data, 
		epochs=100, 
		callbacks=[early_stopping, model_checkpoint_callback, tensorboard_callback],
		validation_split=0.15, 
		batch_size=4, 
		shuffle=True)

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
reconstructed_data = autoencoder.predict(test_data)
mse = np.mean(np.square(test_data - reconstructed_data))
cossim = cosine_similarity(test_data, reconstructed_data)
print(f"Mean Squared Error: {mse}, Cosine Similarity: {cossim}")

#========================================
# Download the model
#========================================
# Save the entire model as a `.keras` zip archive.
decoder.save('cpm_synthetic_data_model.keras')
