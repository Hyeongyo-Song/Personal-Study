import cv2

def onthreshold(pos):
    _, result = cv2.threshold(image, pos, 255, cv2.THRESH_BINARY)
    cv2.imshow(title, result)

title = "threshold"
image = cv2.imread("ironman_01.jpg", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Original", image)
cv2.namedWindow(title)
cv2.createTrackbar("th", title, 0, 255, onthreshold)
cv2.setTrackbarPos("th", title, 100)
cv2.waitKey(0)