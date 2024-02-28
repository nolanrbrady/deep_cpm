import pandas as pd
import numpy as np
import os
import scipy.io
from scipy.stats import pearsonr
import re
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Attention, Concatenate, Lambda, Masking, Conv1D, Conv1DTranspose, Cropping1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_absolute_error
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
def cosine_similarity(matrix1, matrix2):
	# Reshape matrices to 2D (flattening time series and features) for cosine similarity
	flat_matrix1 = matrix1.reshape(matrix1.shape[0], -1)
	flat_matrix2 = matrix2.reshape(matrix2.shape[0], -1)

	# Compute cosine similarity for each pair of corresponding vectors
	similarity = cosine_similarity(flat_matrix1, flat_matrix2)
				    
	return np.diag(similarity) 

def pearson_correlation(matrix1, matrix2):
	correlations = []
	for i in range(matrix1.shape[0]):
		for j in range(matrix1.shape[2]):
			corr, _ = pearsonr(matrix1[i, :, j], matrix2[i, :, j])
			correlations.append(corr)
			return np.mean(correlations)

def calculate_mae(original_data, reconstructed_data):
	    return mean_absolute_error(original_data.flatten(), reconstructed_data.flatten())
    
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

time_steps = train_data.shape[1]
num_features = train_data.shape[2]

input_shape = (time_steps, num_features)

print("Input Shape: ", input_shape)
print(train_data.shape[0], train_data.shape[1], train_data.shape[2])
# should give shape (65, 36209, 268)

# Create the encoder
encoder_inputs = Input(shape=input_shape, name='encoder_input')
encoder_conv1 = Conv1D(filters=256, kernel_size=3, activation='relu', strides=2)(encoder_inputs)
encoder_conv2  = Conv1D(filters=128, kernel_size=3, activation='relu', strides=2)(encoder_conv1)
encoder_conv3  = Conv1D(filters=64, kernel_size=3, activation='relu', strides=2)(encoder_conv2)
encoder_conv4 = Conv1D(filters=32, kernel_size=3, activation='relu', strides=2)(encoder_conv3)

#encoder = Model(encoder_inputs, [encoder_conv4, encoder_conv3, encoder_conv2, encoder_conv1], name='encoder')
encoder = Model(encoder_inputs, encoder_conv4, name='encoder')

# Create the decoder
decoder_input_shape = (2262, 32)
decoder_input = Input(shape=decoder_input_shape)

# Get the encoder outputs for skip connections
#encoder_outputs = encoder(encoder_inputs)
#encoder_conv4_output, encoder_conv3_output, encoder_conv2_output, encoder_conv1_output = encoder_outputs

decoder_conv1 = Conv1DTranspose(filters=64, kernel_size=3, activation='relu', strides=2)(decoder_input)
decoder_conv2 = Conv1DTranspose(filters=128, kernel_size=3, activation='relu', strides=2)(decoder_conv1)
decoder_conv3 = Conv1DTranspose(filters=256, kernel_size=3, activation='relu', strides=2, output_padding=1)(decoder_conv2)
decoder_conv4 = Conv1DTranspose(filters=num_features, kernel_size=3, activation='sigmoid', strides=2)(decoder_conv3)
#cropped_output = Cropping1D(cropping=(0,decoder_conv4.shape[1] - input_shape[0]))(decoder_conv4)

decoder = Model(decoder_input, decoder_conv4, name='decoder')

# Build Autoencoder model
autoencoder_input = Input(shape=input_shape, name='autoencoder_input')
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = Model(autoencoder_input, decoded_img, name='autoencoder')

autoencoder.compile(optimizer='RMSprop', loss='mse')

autoencoder.summary()
encoder.summary()
decoder.summary()

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
		epochs=5, 
		callbacks=[early_stopping, model_checkpoint_callback, tensorboard_callback],
		validation_split=0.15, 
		batch_size=16, 
		shuffle=True)

print("Decoder Input shape: ", decoder_input_shape)
# Generate synthetic data (example using random noise as input)
noise = np.random.normal(size=(50, decoder_input_shape[0], decoder_input_shape[1]))
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
pearsons = pearson_correlation(test_data, reconstructed_data)
mae = calculate_mae(test_data, reconstructed_data)
#cossim = cosine_similarity(test_data, reconstructed_data)
cossim = "pending"
print(f"Mean Squared Error: {mse}, Mean Absolute Error: {mae}, Pearsons: {pearsons}")

#========================================
# Download the model
#========================================
# Save the entire model as a `.keras` zip archive.
decoder.save('cpm_synthetic_data_model.keras')
