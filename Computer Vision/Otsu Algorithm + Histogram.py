# 오츠 알고리즘은 모든 이미지에 대해 최적의 임계값을 찾을 순 없고, Bimodal(다봉분포)이미지에서 적절한 임계값을 찾을 수 있다.

import cv2
from matplotlib import pyplot as plt

title = "threshold"
image = cv2.imread("ironman_01.jpg", cv2.IMREAD_GRAYSCALE)

myth = 171
_, th = cv2.threshold(image, myth, 255, cv2.THRESH_BINARY)
ret, th_otsu = cv2.threshold(image, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

images = {"Original": image, "th : %d" % myth: th, "otsu : %d" % ret: th_otsu}

fig1 = plt.figure(figsize=(6, 8))
for i, (key, value) in enumerate(images.items()):
    plt.subplot(3, 1, i+1)
    plt.title(key)
    plt.imshow(value, cmap="gray")
    plt.xticks([])
    plt.yticks([])

fig2 = plt.figure()
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
plt.title("Histogram")
plt.plot(hist, color="black")
plt.annotate("Otsu", xy=(ret, 0), xytext=(ret, 2000), arrowprops=dict(facecolor="black", arrowstyle="->"))
plt.annotate("myth", xy=(myth, 0), xytext=(myth-20, 4000), arrowprops=dict(facecolor="black", arrowstyle="->"))

plt.show()




