import os
import cv2
import sys
import random
sys.path.insert(0, "./ppocr/data/imaug")

from rec_img_aug import *
from text_image_aug.augment import tia_perspective, tia_stretch, tia_distort
# from .rec_img_aug import blur
root_path='/media/ex/HDD/cmh/trainpic/pastepic/'
save_path='/media/ex/HDD/cmhnew/augpic/'

class Config:
    """
    Config
    """

    def __init__(self, use_tia):
        self.anglex = random.random() * 30
        self.angley = random.random() * 15
        self.anglez = random.random() * 10
        self.fov = 42
        self.r = 0
        self.shearx = random.random() * 0.3
        self.sheary = random.random() * 0.05
        self.borderMode = cv2.BORDER_REPLICATE
        self.use_tia = use_tia

    def make(self, w, h, ang):
        """
        make
        """
        self.anglex = random.random() * 5 * flag()
        self.angley = random.random() * 5 * flag()
        self.anglez = -1 * random.random() * int(ang) * flag()
        self.fov = 42
        self.r = 0
        self.shearx = 0
        self.sheary = 0
        self.borderMode = cv2.BORDER_REPLICATE
        self.w = w
        self.h = h

        self.perspective = False
        self.stretch = False
        self.distort = False

        self.crop = True
        self.affine = False
        self.reverse = False
        self.noise = True
        self.jitter = False
        self.blur = False
        self.color = False
    i=0
    prob=1

    for file in os.listdir(root_path):

        img=cv2.imread(root_path+file)
        # img1=rec_img_aug.warp(img,3)
        h, w, _ = img.shape
        new_img = img
        # 
        if 0<=i<=250000:
            img_height, img_width = img.shape[0:2]
            if random.random() <= prob and img_height >= 20 and img_width >= 20:
                new_img = tia_distort(new_img, random.randint(3, 6))

        if 250000<=i<=500000:
            if random.random() <= prob:
                new_img = 255 - new_img

        if 500000<=i<=750000:
            if random.random() <= prob:
                new_img = add_gasuss_noise(new_img)

        if 750000<=i<=1000000:
            img_height, img_width = img.shape[0:2]
            if random.random() <= prob and img_height >= 20 and img_width >= 20:
                new_img = get_crop(new_img)

        if 1000000<=i<=1250000:
            if random.random() <= prob:
                new_img = blur(new_img)
        cv2.imwrite(save_path+'aug1'+file,new_img)
        i=i+1