# 영상을 검은색과 흰색만으로 표현하는 것이 Thresholding입니다.
# 영상의 화소값을 임계값을 기준으로 2개 값(0,255)으로 분리합니다.
# 원하는 피사체의 모양을 좀더 정확히 구분하고 판단하기 위해 사용합니다.
# cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)
# THRESH_BINARY, THRESH_BINARY_INV, THRESH_TOZERO, THRESH_TOZERO_INV, THREST_TRUNC

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/gray_gradient.jpg", cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
_, binaryinv = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
_, trunc = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
_, zero = cv2.threshold(image, 127, 255 , cv2.THRESH_TOZERO)
_, zeroinv = cv2.threshold(image, 127, 255 , cv2.THRESH_TOZERO_INV)

images = {"origin":image, "BINARY":binary, "BINARY_INV":binaryinv,"TRUNC":trunc,"TOZERO":zero,"TOZERO_INV":zeroinv}

for i, (key, value) in enumerate(images.items()):
    plt.subplot(2, 3, i+1)
    plt.title(key)
    plt.imshow(value, cmap="gray")
    plt.xticks([])
    plt.yticks([])
plt.show()