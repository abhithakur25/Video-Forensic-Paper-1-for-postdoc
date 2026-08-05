from dataclasses import dataclass

from scipy.stats import skew, kurtosis
from termcolor import cprint
import numpy as np
from tqdm import tqdm
from typing import List
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import numpy as np
from keras.applications.resnet import ResNet101
from keras.applications.vgg16 import VGG16
from keras.models import Model
import tensorflow as tf
from SubFunctions.GradCAM import GradCAM
from SubFunctions.LDZP import LocalDirectionalZigZagPattern
resnet = ResNet101()
vgg16 = VGG16()

from skimage.feature import greycomatrix, greycoprops


@dataclass
class FeatureExtraction(object):
    def __init__(self, video: List):
        self.video = video

    @staticmethod
    def grad_cam(image: np.ndarray) -> np.ndarray:
        cprint('\n')
        cprint("[⚠️] Grand Cam based Deep Flow map ", color='grey', on_color='on_yellow')
        cprint("================================", color='blue')
        preprocess = tf.keras.Sequential([
            tf.keras.layers.Resizing(224, 224),
            tf.keras.layers.Rescaling(1 / 127.5, -1),
        ])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        device = 'cpu'
        layer_ = 'out_relu'
        if device == 'cuda' and len(tf.config.list_physical_devices('GPU')) == 0:
            raise ValueError('There is no cuda !!!')

        model = tf.keras.applications.MobileNetV2(classifier_activation=None)

        cam_obj = GradCAM(model, device, preprocess, layer_)
        # output is tf Tensor, overlay is ndarray
        _, overlay = cam_obj.get_heatmap(image)

        return overlay

    @staticmethod
    def get_neighbour(image, x, y):  # comparing bit with threshold value of centre pixel
        try:
            neighbour = image[x][y]
            return neighbour
        except:
            return 0

    def getting_statistical_values(self, image, x, y):
        neighbor1 = self.get_neighbour(image, x - 1, y + 1)
        neighbor2 = self.get_neighbour(image, x, y + 1)
        neighbor3 = self.get_neighbour(image, x + 1, y + 1)
        neighbor4 = self.get_neighbour(image, x + 1, y)
        neighbor5 = self.get_neighbour(image, x + 1, y - 1)
        neighbor6 = self.get_neighbour(image, x, y - 1)
        neighbor7 = self.get_neighbour(image, x - 1, y - 1)
        neighbor8 = self.get_neighbour(image, x - 1, y - 1)

        neighbor_array = np.array([neighbor1, neighbor2, neighbor3, neighbor4,
                                   neighbor5, neighbor6, neighbor7, neighbor8])

        mean = neighbor_array.mean()
        variance = neighbor_array.var()
        std_deviation = neighbor_array.std()
        skew_value = skew(neighbor_array, axis=0, bias=True)
        kurtosis_value = kurtosis(neighbor_array, axis=0, bias=True)

        return [mean, variance, std_deviation, skew_value, kurtosis_value]

    def statistical_features(self, image):

        cprint("[⚠️] Getting Statistical Feature ", color='grey', on_color='on_yellow')
        cprint("================================", color='blue')

        if len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 128))

        m, n = image.shape

        # Finding the mean value,variance value,standard deviation value,skew value ,kurtosis value
        mean_image = np.zeros((m, n))
        variance_image = np.zeros((m, n))
        std_image = np.zeros((m, n))
        skew_image = np.zeros((m, n))
        kurtosis_image = np.zeros((m, n))

        # converting image to lbp
        for i in range(0, m):
            for j in range(0, n):
                [mean, variance, std_deviation, skew_value, kurtosis_value] = self.getting_statistical_values(image, i,
                                                                                                              j)

                mean_image[i][j] = mean
                variance_image[i][j] = variance
                std_image[i][j] = std_deviation
                skew_image[i][j] = skew_value
                kurtosis_image[i][j] = kurtosis_value

        statistical_image = np.zeros(shape=(m, n, 5))
        statistical_image[:, :, 0] = mean_image
        statistical_image[:, :, 1] = variance_image
        statistical_image[:, :, 2] = std_image
        statistical_image[:, :, 3] = skew_image
        statistical_image[:, :, 4] = kurtosis_image

        return statistical_image

    def resnet_statistical(self, image) -> np.ndarray:
        cprint('\n')
        cprint("[⚠️] Hybrid Resnet 101 based statistical features ", color='grey', on_color='on_yellow')
        cprint("================================", color='blue')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        resnet_model = Model(inputs=resnet.inputs, outputs=resnet.layers[2].output)
        outputs = np.squeeze(resnet_model.predict(image))
        outputs = cv2.resize(np.mean(outputs, axis=2), (128, 128))
        outputs = self.statistical_features(outputs)
        return outputs


    @staticmethod
    def vgg_ldzp(image) -> np.ndarray:
        cprint('\n')
        cprint("[⚠️] Getting Deep VGG-16 flow map ", color='grey', on_color='on_yellow')
        cprint("================================", color='blue')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        vgg16_model = Model(inputs=vgg16.inputs, outputs=vgg16.layers[2].output)
        outputs = np.squeeze(vgg16_model.predict(image))
        outputs = cv2.resize(np.mean(outputs, axis=2), (128, 128))
        outputs = LocalDirectionalZigZagPattern(outputs.astype(np.uint8)).get_ldzp()
        return outputs


    @staticmethod
    def object_flow_features(image) -> np.ndarray:
        cprint('\n')
        cprint("[⚠️] Object Flow Features ", color='grey', on_color='on_yellow')
        cprint("================================", color='blue')
        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Parameters for Shi-Tomasi corner detection
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        # Get the frame size from the input image
        frame_height, frame_width = image.shape[:2]
        # Convert the input image to grayscale
        old_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect corners in the first frame
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        if p0 is None:
            p0 = np.random.randint(150.0, 400.0, (5, 1, 2)).astype('float32')
        else:
            p0 = p0
        # Create a mask image for drawing purposes
        mask = np.zeros_like(image)
        # Simulate the next frame by slightly shifting the input image
        next_frame = np.roll(image, 1, axis=1)
        frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # Draw the tracks
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[j].tolist(), 2)
            next_frame = cv2.circle(next_frame, (int(a), int(b)), 5, color[j].tolist(), -1)
        img_with_flow = cv2.add(next_frame, mask)
        return img_with_flow







    def get_features(self) -> np.ndarray:

        Features = []
        for frame in tqdm(self.video, desc='Getting Features from video '):
            gradcam = self.grad_cam(frame)
            resnet_features = self.resnet_statistical(frame)
            vgg_features = self.vgg_ldzp(frame)
            flow = self.object_flow_features(frame)

            Features.append(np.concatenate([gradcam, resnet_features, np.expand_dims(vgg_features, axis=-1), flow], axis=-1))

        return np.array(Features)

    def get_features1(self) -> np.ndarray:

        Features = np.zeros(shape=(128, 128, len(self.video)))
        for i in tqdm(range(len(self.video)), desc='Getting Features from video '):
            feat = cv2.cvtColor(self.video[i], cv2.COLOR_BGR2GRAY)
            feat = feat / np.max(feat)
            feat = cv2.resize(feat, (128, 128))
            Features[:, :, i] = feat
        return Features
    
    @staticmethod
    def glcm_features(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Gray-Level Co-occurrence Matrix (GLCM) features
        glcm_feature = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
        # parameters
        distance = [5]
        angles = [0]
        level = 256
        symmetric = True
        normed = True
        glcm_feat = greycomatrix(img, distance, angles, level, symmetric=symmetric, normed=normed)
        glcm_props = [property for name in glcm_feature for property in greycoprops(glcm_feat, name)[0]]
        glcm_props = np.array(glcm_props)
        return glcm_props

    @staticmethod
    def stat_feature(data):
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        mean1 = np.mean(data)  # mean
        stdev = np.std(data)  # standard deviation
        var1 = np.var(data)  # variance
        median1 = np.median(data)  # median
        skew1 = skew(data.flatten(), axis=0, bias=False)  # skewness
        kurtosis1 = kurtosis(data.flatten(), axis=0, bias=False)  # kurtosis
        feat = np.hstack(
            (mean1, stdev, var1, median1, skew1, kurtosis1))
        return feat

    def get_features2(self) -> np.ndarray:
        Features = np.zeros(shape=(len(self.video), 12))
        for i in tqdm(range(len(self.video)), desc='Getting Features from video '):
            glcm_feat = self.glcm_features(self.video[i])
            stat_feat = self.stat_feature(self.video[i])
            feat = np.concatenate([glcm_feat, stat_feat], axis=0)
            Features[i, :] = feat
        return Features



