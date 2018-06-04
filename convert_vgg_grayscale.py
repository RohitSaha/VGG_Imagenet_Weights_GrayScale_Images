import os
import keras
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from keras.applications.vgg16 import VGG16
from keras.models import load_model, Model
from keras.layers import Input, Conv2D, MaxPooling2D

model = VGG16(weights='imagenet', include_top=False)

# Block1_conv1 weights are of the format [3, 3, 3, 64] -> this is for RGB images
# For grayscale, format should be [3, 3, 1, 64]. Weighted average of the features has to be calculated across channels.
# RGB weights: Red 0.2989, Green 0.5870, Blue 0.1140

# getting weights of block1 conv1.
block1_conv1 = model.get_layer('block1_conv1').get_weights()
weights, biases = block1_conv1

# :weights shape = [3, 3, 3, 64] - (0, 1, 2, 3)
# convert :weights shape to = [64, 3, 3, 3] - (3, 2, 0, 1)
weights = np.transpose(weights, (3, 2, 0, 1))


kernel_out_channels, kernel_in_channels, kernel_rows, kernel_columns = weights.shape

# Dimensions : [kernel_out_channels, 1 (since grayscale), kernel_rows, kernel_columns]
grayscale_weights = np.zeros((kernel_out_channels, 1, kernel_rows, kernel_columns))

# iterate out_channels number of times
for i in range(kernel_out_channels):

	# get kernel for every out_channel
	get_kernel = weights[i, :, :, :]

	temp_kernel = np.zeros((3, 3))

	# :get_kernel shape = [3, 3, 3]
	# axis, dims = (0, in_channel), (1, row), (2, col)

	# calculate weighted average across channel axis
	in_channels, in_rows, in_columns = get_kernel.shape

	for in_row in range(in_rows):
		for in_col in range(in_columns):
			feature_red = get_kernel[0, in_row, in_col]
			feature_green = get_kernel[1, in_row, in_col]
			feature_blue = get_kernel[2, in_row, in_col]

			# weighted average for RGB filter
			total = (feature_red * 0.2989) + (feature_green * 0.5870) + (feature_blue * 0.1140)

			temp_kernel[in_row, in_col] = total


	# :temp_kernel is a 3x3 matrix [rows x columns]
	# add an axis at the end to specify in_channel as 1

	# 2 ways of doing this,

	# First: Add axis directly at the end of :temp_kernel to make its shape: [3, 3, 1], but this might be 
	# an issue when concatenating all feature maps

	# Second: Add axis at the start of :temp_kernel to make its shape: [1, 3, 3] which is [in_channel, rows, columns]
	temp_kernel = np.expand_dims(temp_kernel, axis=0)

	# Now, :temp_kernel shape is [1, 3, 3]

	# Concat :temp_kernel to :grayscale_weights along axis=0
	grayscale_weights[i, :, :, :] = temp_kernel

# Dimension of :grayscale_weights is [64, 1, 3, 3]
# In order to bring it to tensorflow or keras weight format, transpose :grayscale_weights

# dimension, axis of :grayscale_weights = (out_channels: 0), (in_channels: 1), (rows: 2), (columns: 3)
# tf format of weights = (rows: 0), (columns: 1), (in_channels: 2), (out_channels: 3)

# Go from (0, 1, 2, 3) to (2, 3, 1, 0)
grayscale_weights = np.transpose(grayscale_weights, (2, 3, 1, 0)) # (3, 3, 1, 64)

# combine :grayscale_weights and :biases
new_block1_conv1 = [grayscale_weights, biases]


# Reconstruct the layers of VGG16 but replace block1_conv1 weights with :grayscale_weights

# get weights of all the layers starting from 'block1_conv2'
vgg16_weights = {}
for layer in model.layers[2:]:
	if "conv" in layer.name:
		vgg16_weights["1024_" + layer.name] = model.get_layer(layer.name).get_weights()

del model


# Custom build VGG16
input = Input(shape=(1024, 1024, 1), name='1024_input')
# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(1024, 1024, 1), data_format="channels_last", name='1024_block1_conv1')(input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='1024_block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='1024_block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='1024_block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='1024_block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='1024_block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='1024_block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='1024_block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='1024_block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='1024_block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='1024_block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='1024_block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='1024_block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='1024_block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='1024_block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='1024_block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='1024_block5_conv3')(x)
x = MaxPooling2D((8, 8), strides=(8, 8), name='1024_block5_pool')(x)

base_model = Model(inputs=input, outputs=x)

base_model.get_layer('1024_block1_conv1').set_weights(new_block1_conv1)
for layer in base_model.layers[2:]:
	if 'conv' in layer.name:
		base_model.get_layer(layer.name).set_weights(vgg16_weights[layer.name])

base_model.summary()

#print base_model.get_layer('block3_conv2').get_weights()
base_model.save('vgg_grayscale_1024.hdf5')
