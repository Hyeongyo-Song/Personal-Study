# 미디언 블러링을 이용한 점잡음 제거

import cv2
import numpy as np

def salt_pepper_noise(img, n):
    h, w = img.shape[:2]
    x,y = np.random.randint(0,w,n), np.random.randint(0,h,n)
    addnoise = img.copy()
    for (x,y) in zip(x, y):
        addnoise[y, x] = 0 if np.random.rand() < 0.5 else 255
    return addnoise

image = cv2.imread("ironman_01.jpg", cv2.IMREAD_COLOR)
noiseimage = salt_pepper_noise(image, 100)
