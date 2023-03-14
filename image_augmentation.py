import numpy as np
import skimage.io
import skimage.transform
import cv2
import os
from math import sin, cos     
# Define an image augmentor
def augmentor(image, label):
    """Do image shape and color augmentation.
    """
    n_channels = image.shape[0]
 
    # Shape augmentation
    if np.random.uniform(0, 1) < 0.7:
        # Do 70% projection transform
        scale = np.random.uniform(0.8, 1.25)
        aspect_ratio = np.random.uniform(0.8, 1.2)
        rotation = np.random.uniform(-0.2, 0.2)
        translationX = np.random.uniform(-30, 30)
        translationY = np.random.uniform(-30, 30)
        g = np.random.uniform(-0.002, 0.002)
        h = np.random.uniform(-0.002, 0.002)
        matrix = np.array([[math.cos(rotation) * scale * aspect_ratio, -math.sin(rotation), translationX],
                          [math.sin(rotation), math.cos(rotation) * scale / aspect_ratio, translationY],
                          [g, h, 1]])
        tform = skimage.transform.ProjectiveTransform(matrix=matrix)
        image_aug = np.zeros_like(image, dtype=np.float)
        label_aug = np.zeros_like(label, dtype=np.int)
        for ch in range(n_channels):
            image_aug[ch, :, :] = skimage.transform.warp(image[ch, :, :], tform, preserve_range=True)
        label_aug[:, :] = skimage.transform.warp(label[:, :], tform, preserve_range=True, order=0)

        image = image_aug
        label = label_aug
    if np.random.uniform(0, 1) < 0.5:
        # Do 50% vertical flipping
        image = image[:, ::-1, :]
        label = label[::-1, :]
    if np.random.uniform(0, 1) < 0.5:
        # Do 50% horizontal flipping
        image = image[:, :, ::-1]
        label = label[:, ::-1]

    # Color augmentation
    # 1) add a global shifting for all channels
    image = image + np.random.randn(1)[0] * 0.3

    # 2) add a shifting & variance for each channel
    for ch in range(n_channels):
        image[ch, :, :] = image[ch, :, :] * np.clip(np.random.randn(1)[0] * 0.2 + 1, 0.8, 1.2) + np.random.randn(1)[0] * 0.2
     
    return image, label
