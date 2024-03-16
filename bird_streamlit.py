# Importing necessary libraries
# First, we import the libraries we need for this project, including Streamlit for building the web app,
# NumPy for numerical operations,
# Keras for loading pre-trained models, PIL for handling images, and Pandas for working with CSV files.
import streamlit as st  # For building the web app
import numpy as np  # For numerical operations
from keras.models import load_model  # For loading the pre-trained model
from PIL import Image  # For handling images
import pandas as pd  # For working with CSV files

# Next, we define two functions: load_pretrained_model and load_csv. The load_pretrained_model function
# loads our pre-trained model from the 'BC.h5' file,
# while the load_csv function reads the 'birds.csv' file and extracts class labels and IDs.
# Function to load the pre-trained model
@st.cache(allow_output_mutation=True)  # Cache the model to speed up the app
def load_pretrained_model():
    # Load the pre-trained model from the specified file path
    model = load_model('InceptionV3_model.h5')
    return model

# Function to load the CSV file and extract class names and IDs
@st.cache  # Cache the CSV data to speed up the app
def load_csv(csv_file):
    # Read the CSV file into a pandas DataFrame
    birds_df = pd.read_csv(csv_file)
    # Extract the class labels and IDs from the DataFrame and convert them to lists
    class_labels = birds_df['labels'].tolist()
    class_ids = birds_df['class id'].tolist()
    return class_labels, class_ids

# Main function to run the Streamlit app
# In the main function, we create the Streamlit app. We set the title, provide a file uploader for users
# to upload images, and implement the prediction logic. This logic involves loading the uploaded image,
# making predictions using the pre-trained model, and displaying the predicted label, class ID, and class index.
def main():
    # Set the title of the app
    st.title("Bird Species Prediction")
    # Write a message to instruct users to upload an image
    st.write("Upload an image of a bird to predict its species.")

    # Allow users to upload an image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Check if an image has been uploaded
    if uploaded_file is not None:
        # Display the uploaded image on the app
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load the pre-trained model
        model = load_pretrained_model()

        # Load class labels and IDs from the CSV file
        class_labels, class_ids = load_csv('C:/Users/Dell/PycharmProjects/python/bird species/birds.csv')

        # Resize and preprocess the uploaded image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_input = img_array.reshape(1, 224, 224, 3)

        # Make predictions using the model
        predictions = model.predict(img_input)

        # Get the predicted class
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        predicted_class_id = class_ids[predicted_class]

        # Display the predicted label and class ID on the app
        st.write(f"Predicted Label: {predicted_label}")
        st.write(f"Predicted Class ID: {predicted_class_id}")

        # Display the predicted class index
        st.write(f"Predicted Class: {predicted_class}")

# Check if the script is being run directly
# Finally, we check if the script is being run directly and call the main function to run the Streamlit app.
# This will launch our web application, allowing users to upload bird images and get predictions in real-time
if __name__ == "__main__":
    # Call the main function to run the Streamlit app
    main()
