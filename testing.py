import numpy as np
from keras.models import load_model
from PIL import Image
import pandas as pd

# Load the pre-trained model
model = load_model('BC.h5')

# Load your CSV file to extract class names and IDs
birds_df = pd.read_csv('C:/Users/Dell/PycharmProjects/python/bird species/birds.csv')
class_labels = birds_df['labels'].tolist()
class_ids = birds_df['class id'].tolist()

# Load and resize the image
img_path = 'C:/Users/Dell/PycharmProjects/python/bird species/test/ABBOTTS BABBLER/5.jpg'
img = Image.open(img_path)
img = img.resize((224, 224))

# Convert the image to a numpy array
img_array = np.array(img)

# Reshape the array to match the model's input shape
img_input = img_array.reshape(1, 224, 224, 3)

# Make predictions
predictions = model.predict(img_input)

# Get the predicted class
predicted_class = np.argmax(predictions)
print('Predicted class:', predicted_class)

# Get the predicted label and class ID from the CSV file
predicted_label = class_labels[predicted_class]
predicted_class_id = class_ids[predicted_class]

print('Predicted label:', predicted_label)
print('Predicted class ID:', predicted_class_id)
