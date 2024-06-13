from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils import load_img, img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims
import os
import tensorflow as tf

#---------------------------------#
# Load the image
#---------------------------------#
data_dir = '../data/Pictures/'
list_of_image = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
selected_img = list_of_image[0]
img = load_img(data_dir + selected_img, target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)

#plot imag
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# ax.imshow(img[0])
# ax.axis('off')
# plt.show()

#-------------------------------#
# Load the VGG model
#-------------------------------#
# load the model
model = VGG16()

# summarize feature map shapes
# for i in range(len(model.layers)):
#     layer = model.layers[i]
#     # check for convolutional layer
#     if 'conv' not in layer.name:
#         continue
#     # summarize output shape
#     print(i, layer.name, layer.output.shape)

# redefine model to output right after the first hidden layer
feature_model = Model(inputs=model.inputs, outputs=model.layers[6].output)

# Print the summary of the model (to see the output shape the model, number of parameters, etc.)
# feature_model.summary()

# get feature map for first hidden layer
with tf.device('/cpu:0'):
    # get feature map for first hidden layer
    feature_maps = feature_model.predict(img)

# --------------------------------- #
# Plot the feature maps
print(feature_maps.shape[-1])

fig2, ax2 = plt.subplots(8, 8, figsize=(20, 20))
for i in range(8):
    for j in range(8):
        ax2[i, j].imshow(feature_maps[0, :, :, i*8+j], cmap='gray')
        ax2[i, j].axis('off')
#fig set title (name of the last layer)
plt.suptitle('Convolution layer : ' + feature_model.layers[-1].name)
plt.show()
#save image example directory as VGG16_feature_map_bird.jpg
#fig2.savefig('../data/Pictures/VGG16_feature_map_bird.jpg')





