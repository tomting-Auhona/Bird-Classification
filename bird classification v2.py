import os

# 1. Counting the Number of Bird Groups:
# "We start by counting the number of bird groups present in our training directory.
# This gives us an idea of how many different species we're working with."

# Count the number of bird groups in the training directory
num_of_bird_groups = len(os.listdir("C:/Users/Dell/PycharmProjects/python/bird species/train"))
print(num_of_bird_groups)

import pathlib
import numpy as np

# Set the data directory path
data_dir = pathlib.Path("C:/Users/Dell/PycharmProjects/python/bird species/train")

# 2. Getting Class Names: "Next, we gather the class names from the subdirectories within our training
# data directory.
# These class names represent the various bird species we'll be classifying."

# Get class names from subdirectories
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
print(class_names)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# 3. Viewing a Random Image: "Here, we define a function to view a random image from a specified target directory
# and class. " \
# This allows us to visually inspect our data and understand what our model will be learning."

# Function to view a random image from a specified target directory and class
def view_random_image(target_dir, target_class):
    # Set the target folder
    target_folder = target_dir + target_class

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)

    # Read and plot the image
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")

    print(f"Image shape: {img.shape}")

    return img

# View a random image
img = view_random_image(target_dir="C:/Users/Dell/PycharmProjects/python/bird species/train/",
                        target_class="VICTORIA CROWNED PIGEON")

import tensorflow as tf

# Check the shape of the image
img.shape

import matplotlib.pyplot as plt
import pathlib, os, random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow as tf

from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential

# Define directories for training, testing, and validation
train_dir = 'C:/Users/Dell/PycharmProjects/python/bird species/train'
test_dir = 'C:/Users/Dell/PycharmProjects/python/bird species/test'
val_dir = 'C:/Users/Dell/PycharmProjects/python/bird species/valid'

# Loading and Preprocessing Image Data: "We set up image data generators for training, testing, and validation.
# These generators preprocess the image data
# and generate batches of augmented images, which are then fed into our neural network models for training."

# Set up image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training, testing, and validation data
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="categorical")

test_data = test_datagen.flow_from_directory(directory=test_dir,
                                             batch_size=32,
                                             target_size=(224, 224),
                                             class_mode="categorical")

val_data = valid_datagen.flow_from_directory(directory=val_dir,
                                             batch_size=32,
                                             target_size=(224, 224),
                                             class_mode="categorical")

test_data

import numpy as np
import tensorflow as tf
from keras.applications import InceptionV3, ResNet50, EfficientNetB0, MobileNetV2, DenseNet121
from keras.optimizers import Adam
import tensorflow_addons as tfa
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from keras.models import Model
from keras.layers import Dense

# 5. Creating and Training Transfer Learning Models: "Now comes the exciting part! We create and train transfer
# learning models using pre-trained architectures such as InceptionV3, ResNet50, EfficientNetB0, MobileNetV2,
# and DenseNet121. These models are powerful deep learning architectures that have been pre-trained on large
# datasets like ImageNet.
# We leverage their learned features and fine-tune them for our specific bird species classification task."

# Function to create a transfer learning model
def create_model(base_model):
    # Remove the top layer
    base_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # Add a logistic layer with 525 output units (one for each bird species)
    predictions = Dense(525, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=[F1Score(num_classes=525, average='macro'), 'accuracy'])

    return model


base_models = [InceptionV3, ResNet50, EfficientNetB0, MobileNetV2, DenseNet121]

# Dictionary to store models and their names
models = {}

# Loop through base models to create and train transfer learning models
for model_name, base_model in zip(['InceptionV3', 'ResNet50', 'EfficientNetB0', 'MobileNetV2', 'DenseNet121'],
                                  base_models):
    # Create and compile the model
    model = create_model(base_model(weights='imagenet', include_top=False))
    models[model_name] = model

    # Train the model
    history = model.fit(train_data, steps_per_epoch=len(train_data), validation_data=val_data,
                        validation_steps=len(val_data), epochs=3)

    # Optionally, save the model for later use
    model.save(f'{model_name}_model.h5')

# 6. Evaluating Trained Models: "After training, we evaluate the performance of each model using metrics
# such as accuracy, F1 score, classification report, and confusion matrix.
# This gives us insights into how well our models are performing and where they might need improvement."

# Evaluate trained models
for model_name, model in models.items():

    # Generate predictions
    y_prob = model.predict(test_data)
    y_pred = np.argmax(y_prob, axis=1)

    # Extract true labels
    y_true = []

    test_labels = []

    for i in range(len(test_data)):
        test_labels.extend(test_data[i][1])

    # Convert one-hot encoded labels to class indices
    y_true = np.argmax(test_labels, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)

    # Print metrics
    print(f'{model_name} Metrics:')
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1}')
    print('Classification Report:')
    print(report)
    print('Confusion Matrix:')
    print(matrix)
