# -*- coding: utf-8 -*-
"""preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15o7KRaSeP5vRAIBbWxUVi8xWbgTs6eXf
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import scipy.io as sio
from ImageResizer import cropper
from Settings import IMAGE_SIZE

print(tf.version)


normal = sio.loadmat("./checkpoint_data/after_matlab_scripts/NormalRPeaks.mat")
af = sio.loadmat("./checkpoint_data/after_matlab_scripts/AfRPeaks.mat")

normalImages = np.empty([normal["NormalRPeaks"].shape[1], 360, 360])
afImages = np.empty([af["AfRPeaks"].shape[1], 360, 360])

normalImagesCropped = np.empty(
    [normal["NormalRPeaks"].shape[1], IMAGE_SIZE, IMAGE_SIZE]
)
afImagesCropped = np.empty([af["AfRPeaks"].shape[1], IMAGE_SIZE, IMAGE_SIZE])

for i in range(0, normal["NormalRPeaks"].shape[1]):
    print("i")
    print(i)
    x = np.diff(np.array(normal["NormalRPeaks"][0][i][0], dtype=np.int16), 2)
    fig = plt.figure(figsize=(3.6, 3.6))
    plt.plot(x[0:-2], x[1:-1])
    plt.xlim([-500, 500])
    plt.ylim([-500, 500])
    plt.axis("off")
    fig.canvas.draw()  # draw the canvas, cache the renderer

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = image[:, :, 0] / 256
    normalImages[i, :, :] = image

np.save("./checkpoint_data/after_preprocessing/normalImages.npy", normalImages)


for i in range(0, af["AfRPeaks"].shape[1]):
    print("i")
    print(i)
    i_new = i
    x = np.diff(np.array(af["AfRPeaks"][0][i_new][0], dtype=np.int16), 2)
    fig = plt.figure(figsize=(5, 5))
    plt.plot(x[0:-2], x[1:-1])
    plt.xlim([-500, 500])
    plt.ylim([-500, 500])
    plt.axis("off")
    fig.canvas.draw()  # draw the canvas, cache the renderer

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = image[:, :, 0] / 256
    afImages[i, :, :] = image

    print(image.shape)

np.save("./checkpoint_data/after_preprocessing/afImages.npy", afImages)
normalImages = np.empty([normal["NormalRPeaks"].shape[1], 360, 360])
afImages = np.empty([af["AfRPeaks"].shape[1], 360, 360])

normalImagesCropped = np.empty(
    [normal["NormalRPeaks"].shape[1], IMAGE_SIZE, IMAGE_SIZE]
)
afImagesCropped = np.empty([af["AfRPeaks"].shape[1], IMAGE_SIZE, IMAGE_SIZE])

normalImagesCropped = cropper(normalImages, IMAGE_SIZE)
np.save(
    "./checkpoint_data/after_preprocessing/normalImagesCropped.npy", normalImagesCropped
)

afImagesCropped = cropper(afImages, IMAGE_SIZE)
np.save("./checkpoint_data/after_preprocessing/afImagesCropped.npy", afImagesCropped)
