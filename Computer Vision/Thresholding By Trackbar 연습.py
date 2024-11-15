import cv2

def onthreshold(pos):
    _, result = cv2.threshold(Image, pos, 255, cv2.THRESH_BINARY)
    cv2.imshow("Image", Image)

OriginalImage = cv2.imread("ironman_01.jpg", cv2.IMREAD_GRAYSCALE)
Image = cv2.imread("ironman_02.jpg", cv2.IMREAD_GRAYSCALE)

cv2.createTrackbar("Thresholding", "Image", 0, 255, onthreshold)

cv2.imshow("Original Image", OriginalImage)
cv2.imshow("Image", Image)
cv2.waitKey(0)