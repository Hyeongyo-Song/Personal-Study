# 이미지의 각 픽셀에 대하여 개별적인 임계값을 적용하는 방식으로 이진화를 진행하는 것을 적응형 이진화라 한다.
# 이미지를 여러 영역으로 나누고, 각 영역마다 개별 임계값을 적용 후 더하는 방법도 있다.
# 임계값을 전체적으로 적용하는 것을 전역적 적용이라 하고, 영역을 나누어 적용하는 것을 지역적 적용이라 한다.
# Adapted-Mean과 Adapted-Gaussian 방식이 존재.
# 픽셀 : 픽셀을 비교함.
# cv2.adaptiveThreshold(Matrix, Threshold, Method(MEAN,GAUSSIAN), Threshold Type, Block Size, C(차감계수 : 계산한 결과에 차감하여 임계값 결정))

import cv2
from matplotlib import pyplot as plt

bsize = 9
c = 5
image = cv2.imread("sudoku.png", cv2.IMREAD_GRAYSCALE)
ret, th1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bsize, c)
th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bsize, c)

images = {"Original": image, "Otsu : %d" % ret: th1, "Adapted-Mean": th2, "Adapted-Gaussian": th3}
for i, (key, value) in enumerate(images.items()):
    plt.subplot(2, 2, i+1)
    plt.title(key)
    plt.imshow(value, cmap="gray")
    plt.xticks([]), plt.yticks([])
plt.show()
