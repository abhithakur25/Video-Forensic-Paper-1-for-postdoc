from typing import List
import cv2  # OpenCV library for image and video processing
import numpy as np  # Numpy for numerical operations
from tqdm import tqdm  # tqdm for displaying a progress bar


# Define a class that handles video preprocessing
class Preprocessing(object):
    def __init__(self, video: list):
        # Initialize with a list of video frames
        self.video = video

    @staticmethod
    def roi(image: np.ndarray) -> np.ndarray:
        # Method to extract the Region of Interest (ROI) from an image (face detection)

        # Convert the input image to grayscale for better face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load a pre-trained Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier('Temp\\haarcascade_frontalface_alt2.xml')

        # Detect faces in the grayscale frame using the classifier
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # If faces are detected, loop through each face and extract the region of interest (face)
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]  # Crop the face region from the image
            face = cv2.resize(face, (128, 128))
            return face  # Return the cropped face region

        # If no face is detected, the function will return None (optional: could return original image)
        return cv2.resize(image, (128, 128))

    def get_preprocessing(self) -> List:
        # Method to preprocess each frame in the video and extract ROIs (faces)

        roi = []  # Initialize an empty list to store the regions of interest (faces)

        # Iterate through all frames in the video with a progress bar
        for frame in tqdm(self.video, desc='Preprocessing video '):
            # Apply the ROI function to each frame and append the result to the roi list
            roi_ = self.roi(frame)
            roi.append(roi_)

        # Return the list containing all the face regions (ROIs) from the video frames
        return roi
