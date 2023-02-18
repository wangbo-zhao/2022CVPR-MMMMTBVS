import torchvision.transforms as transforms

import torch
import random
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F

im_mean = (124, 116, 104)

im_normalization = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])

def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class ResizeAndPad(object):
    def __init__(self, input_h, input_w, method):
        self.input_h = input_h
        self.input_w = input_w
        self.method = method


    def __call__(self, image):

        image_w, image_h = image.size

        scale = min(self.input_h / image_h, self.input_w / image_w)

        resized_h = int(np.round(image_h * scale))
        resized_w = int(np.round(image_w * scale))

        pad_h = int(np.floor(self.input_h - resized_h) / 2)
        pad_w = int(np.floor(self.input_w - resized_w) / 2)

        image = image.resize((resized_w, resized_h), self.method)
        image = np.array(image) #[h, w, 3]

        if image.ndim > 2:
            new_img = np.zeros((self.input_h, self.input_w, image.shape[2]), dtype=image.dtype)
        else:
            new_img = np.zeros((self.input_h, self.input_w), dtype=image.dtype)

        new_img[pad_h:pad_h+resized_h, pad_w:pad_w+resized_w, ...] = image

        new_img = Image.fromarray(new_img)

        return new_img


