from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image, ImageStat
import numpy as np
import random
import math


class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):

        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width, interpolation=self.interpolation)
        return scale(img)

# -----------------------------
# Trick 2: Random Erasing Augmentation
def decision(probability):
    return random.random() < probability

class RandomErasingAugmentation(object):
    def __init__(self, height, width, interpolation = Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):

        if decision(0.5):
            return img

        while True:
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.02, 0.4) * area
            aspect_ratio = random.uniform(0.3, 3.33)

            # calculating height and width of the rectagle region to erase
            H_e = int(round(math.sqrt(target_area * aspect_ratio)))
            W_e = int(round(math.sqrt(target_area / aspect_ratio)))

            # selecting random point
            x_e = random.randint(0, img.size[0] - W_e)
            y_e = random.randint(0, img.size[1] - H_e)

            print("H_e: {}, W_e: {}, x_e: {}, y_e: {}, height: {}, width: {}".format(H_e, W_e, x_e, y_e, img.size[1], img.size[0]))

            # checking if the rectangle region is inside the image size
            if x_e + W_e <= img.size[0] and y_e + H_e <= img.size[1]:

                # calculating the mean 
                stat = ImageStat.Stat(img)
                mean = [int(elem) for elem in stat.mean]

                # adding the mean
                img = np.asarray(img, dtype = "float32")
                img[y_e : y_e + H_e, x_e : x_e + W_e] = mean

                img = Image.fromarray(img.astype('uint8'), 'RGB')

                return img.resize((self.width, self.height), self.interpolation)
# -----------------------------