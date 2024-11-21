import cv2
import numpy as np

image = cv2.imread("ironman_01.jpg", cv2.IMREAD_COLOR)

kernel3 = np.ones((3,3)) / 9
kernel5 = np.ones((5,5)) / 25

blured3 = cv2.filter2D(image, -1, kernel3, borderType=cv2.BORDER_CONSTANT)
blured5 = cv2.filter2D(image, -1, kernel5, borderType=cv2.BORDER_CONSTANT)

cv2.imshow("Original", image)
cv2.imshow("blured 3", blured3)
cv2.imshow("blured 5", blured5)

cv2.waitKey(0)
cv2.destroyAllWindows()