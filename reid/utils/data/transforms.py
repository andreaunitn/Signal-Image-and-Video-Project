from __future__ import absolute_import

from torchvision.transforms import *
from torchvision.transforms import transforms
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

        doResize = True

        # Resize
        img = img.resize((self.width, self.height), self.interpolation)

        # Padding
        w, h = img.size
        width = w + 2 * 10
        height = h + 2 * 10

        result = Image.new(img.mode, (width, height), (0,0,0))
        result.paste(img, (10,10))

        img = result

        for _ in range(10):

            area = img.size[0] * img.size[1]

            if(doResize):
                target_area = 128*256
                aspect_ratio = 2
            else: 
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

class RandomErasingAugmentation(object):
    def __init__(self, height, width, interpolation = Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):

        if random.uniform(0,1) >= 0.5:
            return img

        while True:
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(0.02, 0.4) * area
            aspect_ratio = random.uniform(0.3, 3.33)

            # calculating height and width of the rectagle region to erase
            H_e = int(round(math.sqrt(target_area * aspect_ratio)))
            W_e = int(round(math.sqrt(target_area / aspect_ratio)))

            # checking if the rectangle region is inside the image size
            if W_e < img.size()[2] and H_e < img.size()[1]:

                # selecting random point
                x_e = random.randint(0, img.size()[1] - H_e)
                y_e = random.randint(0, img.size()[2] - W_e)

                mean=(0.4914, 0.4822, 0.4465)

                if img.size()[0] == 3:
                    img[0, x_e:x_e + H_e, y_e:y_e + W_e] = mean[0]
                    img[1, x_e:x_e + H_e, y_e:y_e + W_e] = mean[1]
                    img[2, x_e:x_e + H_e, y_e:y_e + W_e] = mean[2]
                else:
                    img[0, x_e:x_e + H_e, y_e:y_e + W_e] = mean[0]

                return img
# -----------------------------