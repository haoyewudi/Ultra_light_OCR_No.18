import numpy as np
import cv2
import math

def resize_norm_img_chinese(img, image_shape):
    imgC, imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(32 * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    # if image_shape[0] == 1:
    #     resized_image = resized_image / 255
    #     resized_image = resized_image[np.newaxis, :]
    # else:
    resized_image = resized_image.transpose((2, 0, 1))
    # resized_image -= 0.5
    # resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    padding_im = padding_im.transpose((1, 2, 0))
    return padding_im
if __name__ == '__main__':
    img=cv2.imread('/media/ex/HDD/PaddleOCR-release-2.1/doc/imgs_words/ch/word_1.jpg')
    image_shape=3,32,380
    p=resize_norm_img_chinese(img, image_shape)
    cv2.imwrite('/media/ex/HDD/PaddleOCR-release-2.1/doc/imgs_words/ch/wordnew.jpg',p)