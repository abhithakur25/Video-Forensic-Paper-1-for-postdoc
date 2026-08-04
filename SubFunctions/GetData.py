import glob
import pickle
import random
from dataclasses import dataclass
from typing import List
import cv2
import numpy as np
import tqdm
from termcolor import cprint
from SubFunctions.GetPreprocessing import Preprocessing
from SubFunctions.GetFeatures import FeatureExtraction


@dataclass
class ReadDataset(object):
    def __init__(self, exec: bool = True):
        self.exec = exec

    @staticmethod
    def read_video(filename: str) -> List:
        frames = []

        # Open the video file using OpenCV

        cap = cv2.VideoCapture(filename)

        # Get the total number of frames in the video
        totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Check if the video file is opened successfully

        if not cap.isOpened():
            print("Error opening video file")

        # Loop through each frame in the video

        while cap.isOpened():
            # Read the frame
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

                # Break the loop if 'q' is pressed

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        start_range = 0
        end_range = int(totalNoFrames) - 1
        num_values = 10
        counts_ = np.linspace(start_range, end_range, num_values).astype(int)
        frames = [frames[i] for i in counts_]
        return frames





    def read_data(self) -> dict:

        if self.exec:
            cprint(f"[⁉️] Extracting the Extracted Features and Labels ", color='grey', on_color='on_white')

            path1 = glob.glob("DATASET\\manipulated_sequences\\FaceSwap\\c23\\videos\\*.mp4")
            path2 = glob.glob("DATASET\\original_sequences\\youtube\\c23\\videos\\*.mp4")

            path = path1 + path2
            random.shuffle(path)

            Features = []  # List to store the features
            Features1 = []  # Placeholder for multiple feature sets
            Features2 = []
            Features3 = []
            Features4 = []
            Features5 = []
            Labels = []  # List to store labels

            for filename in tqdm.tqdm(path, desc='Reading videos '):
                video = self.read_video(filename)
                preprocessed = Preprocessing(video).get_preprocessing()
                features1 = FeatureExtraction(preprocessed).get_features1()
                features2 = FeatureExtraction(preprocessed).get_features1()
                features3 = FeatureExtraction(preprocessed).get_features1()
                features4 = FeatureExtraction(preprocessed).get_features2()
                features5 = FeatureExtraction(preprocessed).get_features1()
                features = FeatureExtraction(preprocessed).get_features()

                # Append the extracted features and label (1 for class 'L')
                Features.append(features)
                Features1.append(features1)
                Features2.append(features2)
                Features3.append(features3)
                Features4.append(features4)
                Features5.append(features5)

                if 'original_sequences' in filename:
                    Labels.append(0)

                else:
                    Labels.append(1)

            # Convert the extracted features to numpy arrays
            Features = np.array(Features)
            Features1 = np.array(Features1)
            Features2 = np.array(Features2)
            Features3 = np.array(Features3)
            Features4 = np.array(Features4)
            Features5 = np.array(Features5)
            Labels = np.array(Labels).astype(int)

            # Save the extracted features and labels to a pickle file for future use
            data = {
                'comparative1': Features1,
                'comparative2': Features2,
                'comparative3': Features3,
                'comparative4': Features4,
                'comparative5': Features5,
                'proposed': Features,
                'labels': Labels,
            }

            with open(f'Features\\Features.pkl', 'wb') as f:
                pickle.dump(data, f)

            # Print success message after feature extraction
            cprint(f"[✅] Feature Extraction Done !! ", color='grey', on_color='on_white')

        else:
            # Load the extracted features from the pickle file if `exec` is False
            cprint(f"[⁉️] Loading the Extracted Features and Labels ", color='grey', on_color='on_white')

            with open(f'Features\\Features.pkl', 'rb') as f:
                data = pickle.load(f)

            # Print success message after loading the data
            cprint(f"[✅] Feature Loading Done !! ", color='grey', on_color='on_white')

        return data  # Return the data (features and labels)
