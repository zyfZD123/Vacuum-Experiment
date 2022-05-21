from cgi import test
import matplotlib.pyplot as plt
import cv2

test_img = cv2.imread("./3.png", cv2.IMREAD_UNCHANGED)
# test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
# _, test_img = cv2.threshold(test_img, 127, 255, cv2.THRESH_BINARY)
sample1 = test_img[0 : 58, 6 : 51]
sample2 = test_img[0 : 58, 69 : 114]
sample3 = test_img[0 : 58, 132 : 177]
sample4 = test_img[0 : 58, 195 : 240]
sample5 = test_img[0 : 58, 259 : 304]
cv2.imwrite("4.png", sample1)
cv2.imwrite("5.png", sample2)
cv2.imwrite("6.png", sample3)
cv2.imwrite("7.png", sample4)
cv2.imwrite("8.png", sample5)
plt.imshow(test_img)
plt.show()