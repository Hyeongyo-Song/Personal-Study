import cv2


def onthreshold(pos):
    bsize = cv2.getTrackbarPos("Block size", title)
    c = cv2.getTrackbarPos("C", title)
    if bsize % 2 == 0:
        bsize = bsize - 1
    if bsize < 3:
        bsize = 3
    print("Block size : %d / C : %d" % (bsize, c))
    result = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bsize, c)
    cv2.imshow(title, result)


title = "Adaptive Threshold"
image = cv2.imread("ironman_02.jpg", cv2.IMREAD_GRAYSCALE)

cv2.namedWindow(title)
cv2.createTrackbar("Block size", title, 0, 50, onthreshold)
cv2.createTrackbar("C", title, 0, 50, onthreshold)
cv2.setTrackbarPos("Block size", title, 5)
cv2.setTrackbarPos("C", title, 3)
# onthreshold(0)
cv2.waitKey(0)
