# Import necessary libraries
# First, we import the libraries we'll need for this project. These include Keras for building the model,
# TensorFlow for backend operations, and Matplotlib for plotting the training and validation metrics."
from keras.layers import Input, Lambda, Dense, Flatten  # Layers for building the model
from keras.models import Model  # Model API
from keras.applications.vgg16 import VGG16  # Pre-trained VGG16 model
import tensorflow as tf  # TensorFlow library
from keras.applications.vgg16 import preprocess_input  # Function for preprocessing input images
from keras.preprocessing import image  # Image preprocessing utilities
from keras.models import Sequential  # Sequential model API
import numpy as np  # Numerical operations
from glob import glob  # File path pattern matching
import matplotlib.pyplot as plt  # Plotting library

# Print TensorFlow version
print(tf.__version__)
# Print Keras version
print(tf.keras.__version__)

# Define image size
IMAGE_SIZE = [224, 224]

# Define directories for training, testing, and validation datasets
train_directory = 'C:/Users/Dell/PycharmProjects/python/bird species/train'
test_directory = 'C:/Users/Dell/PycharmProjects/python/bird species/test'
val_directory = 'C:/Users/Dell/PycharmProjects/python/bird species/valid'

# Next, we load a pre-trained VGG16 model. VGG16 is a popular deep learning architecture known for its
# excellent performance in image classification tasks.
# We exclude the top layer because we'll add our custom dense layer for bird species classification.

# Load the pre-trained VGG16 model with weights from ImageNet and exclude the top layer
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# We then add a dense layer with softmax activation for classification on top of the VGG16 base.
# This dense layer will output the probability distribution over the different bird species.
# We compile the model with categorical cross-entropy loss and the Adam optimizer.

# Freeze the weights of the pre-trained VGG16 model so they are not trainable
for layer in vgg.layers:
  layer.trainable = False

# Get the number of output classes
folders = glob('C:/Users/Dell/PycharmProjects/python/bird species/train/*')
num_classes = len(folders)

# Flatten the output of the pre-trained VGG16 model
x = Flatten()(vgg.output)

# Add a dense (fully connected) layer with softmax activation for classification
prediction = Dense(num_classes, activation='softmax')(x)

# Create a new model with VGG16 as the base and the added dense layer
model = Model(inputs=vgg.input, outputs=prediction)

# Display the structure of the model
model.summary()

# Compile the model with categorical cross-entropy loss and Adam optimizer
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Data augmentation is essential for training deep learning models on limited datasets.
# Here, we use Keras' ImageDataGenerator to perform real-time data augmentation on the training images.
# We also preprocess the images by rescaling their pixel values to the range [0, 1].
# Data augmentation and preprocessing for training and testing images
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Now, we train the model using the generated batches of augmented data.
# We specify the number of epochs and steps per epoch.
# After training, we plot the training and validation loss as well as accuracy to evaluate the model's performance.

# Generate batches of augmented training and testing data
training_set = train_datagen.flow_from_directory(train_directory,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_directory,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# Print the number of batches in the training and testing sets
print(len(training_set))
print(len(test_set))

# Once the model is trained, we save it to a file named 'BC.h5'.
# This file will contain the trained weights and architecture of the model,
# allowing us to load and reuse it later."

# Train the model using the generated batches of data
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# Plot training and validation loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

# Save the trained model to a file
model.save('BC.h5')

# Import necessary libraries for loading the saved model and image preprocessing
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array

# Finally, we load the saved model and define a function called 'output' to make predictions on new images.
# This function takes the file path of an image as input, preprocesses it, and then predicts the bird species
# using the trained model.

# Load the saved model
model1 = load_model('./BC.h5',compile=False)

# Get class labels from the training set
lab = training_set.class_indices
# Invert the dictionary to map class indices to labels
lab={k:v for v,k in lab.items()}

# Function to predict the class label of an input image
def output(location):
    # Load and preprocess the input image
    img=load_img(location,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    # Predict the class probabilities for the input image
    answer=model1.predict(img)
    # Get the index of the predicted class with the highest probability
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    # Map the predicted class index to the corresponding label
    res = lab[y]
    return res

# Example usage of the output function with an image path
img= 'C:/Users/Dell/PycharmProjects/python/bird species/valid/BARN OWL/1.jpg'
pic=load_img('C:/Users/Dell/PycharmProjects/python/bird species/valid/BARN OWL/1.jpg',target_size=(224,224,3))
plt.imshow(pic)
output(img)
