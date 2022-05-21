import os
import cv2

i = 1
input = './sample/'
for img in os.listdir(input):
    image = cv2.imread(input+img, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(input+'{i}.png'.format(i = i), image)
    os.remove(input+img)
    i += 1