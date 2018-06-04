# VGG_Imagenet_Weights_GrayScale_Images
Convert VGG imagenet pre-trained weights for grayscale images.

2 methods:

1. Convert your images to grayscale, copy the grayscale channel 2 times to make the image 3-D.

2. Convert the weights of VGG16's first convolutional layer to accomodate gray-scale images.
eg: Dimension of VGG16's block1_conv1 kernel: (3, 3, 3, 64) -> (height, width, in_channels, out_channels). By default, the in_channels correspond to the number of channels yout training images have. Since VGG16 is pre-trained on Imagenet that has RGB images, in_channels is 3. The idea is to extract these weight values, do a weighted-average of the filters (channel wise) and assign these values to block1_conv1_kernel, s.t. dimension becomes (3, 3, 1, 64).

Luminosity formula is used to calculate weighted average:
value: (feature_red * 0.2989) + (feature_green * 0.5870) + (feature_blue * 0.1140)

