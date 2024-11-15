import cv2
from matplotlib import pyplot as plt

title = "threshold"
image = cv2.imread("ironman_01.jpg", cv2.IMREAD_GRAYSCALE)

myth = 171
_, th = cv2.threshold(image, myth, 255, cv2.THRESH_BINARY)
ret, th_otsu = cv2.threshold(image, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

images = {"Original": image, "th : %d" % myth: th, "otsu : %d" % ret: th_otsu}

fig = plt.figure(figsize=(6, 8))
for i, (key, value) in enumerate(images.items()):
    plt.subplot(3, 1, i+1)
    plt.title(key)
    plt.imshow(value, cmap="gray")
    plt.xticks([])
    plt.yticks([])

plt.show()
