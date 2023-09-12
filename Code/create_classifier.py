# Import necessary libraries
import numpy as np
from PIL import Image
import os
import cv2

# Method to train a custom classifier to recognize a face


def train_classifier(name):
    # Read all the images in the custom data-set folder for a specific 'name'
    path = os.path.join(os.getcwd() + "/data/" + name + "/")

    # Initialize lists to store faces, associated IDs, and labels
    faces = []
    ids = []
    labels = []
    pictures = {}

    # Iterate through the files in the custom data-set folder
    for root, dirs, files in os.walk(path):
        pictures = files

    # Process each picture in the folder
    for pic in pictures:
        # Create the full image path
        imgpath = path + pic
        # Open the image using PIL and convert it to grayscale ('L' mode)
        img = Image.open(imgpath).convert('L')
        # Convert the PIL image to a numpy array of unsigned 8-bit integers
        imageNp = np.array(img, 'uint8')
        # Extract the ID from the file name (assumes the ID is before the 'name' in the file name)
        id = int(pic.split(name)[0])
        # Append the face image and its associated ID to their respective lists
        faces.append(imageNp)
        ids.append(id)

    # Convert the 'ids' list to a numpy array
    ids = np.array(ids)

    # Create an LBPH (Local Binary Pattern Histograms) face recognizer
    clf = cv2.face.LBPHFaceRecognizer_create()
    # Train the recognizer using the faces and their associated IDs
    clf.train(faces, ids)
    # Save the trained classifier to a file
    clf.write("./data/classifiers/" + name + "_classifier.xml")

# The function 'train_classifier' reads face images from a specific folder, extracts IDs from file names,
# trains an LBPH face recognizer, and saves the trained model to a file.
