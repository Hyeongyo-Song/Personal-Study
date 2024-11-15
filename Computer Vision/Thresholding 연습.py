import cv2
import numpy as np

image = np.zeros((200,300), np.uint16)
color = 0

for x in range(300):
    for y in range(200):
        color += 1
        image[y,x] = color

_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("image", image)
cv2.imshow("binary", binary)
cv2.waitKey(0)