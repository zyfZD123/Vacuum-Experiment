import cv2
import numpy as np
import sys
import os
import datetime
import xlwt

TEMPLATE_DIR = "../template images/"
VIDEO_DIR = "../videos/"
RESULT_DIR = "../resultData/"
MODEL_DIR = "../"
MAX_FEATURES = 500

"""
#### assisting functions
- `pad_image()`: used in `processImage()`
- `findMin()`: used in `processImage()`
- `drawImages_Gray()`: just for debugging.
- `isValidNum()`: will be called in `readNumber()` to judge whether it reads a valid number.
- `openVideo()`: check if the video exists and open it
- `showVideoInfo()`: show information of video, such as FPS, sampling_rate, etc.
- `initExcelTable()`: initialize a workbook
- `showProgress()`: a simple implementation of a progress bar
- `saveResult()`: it will save the workbook to the current directory.

"""

def pad_image(im, height, width): #(height width) are the target height and width.
    # im = Image.fromarray(im)  # convert "im" from "array" to "PIL Image". Now "im" is 2D. 
    # w, h = im.size  
    # if w>=h*width/height:
    #     h1 = int(h*width/w)
    #     im = im.resize((width, h1),  Image.BILINEAR)
    #     im_new = Image.new(mode='L', size=(width, height), color=0)
    #     im_new.paste(im, ( 0, (height-h1)//2 ) )
    # else:
    #     w1 = int(w*height/h)
    #     im = im.resize((w1, height),  Image.BILINEAR)
    #     im_new = Image.new(mode='L', size=(width, height), color=0)
    #     im_new.paste(im, ( (width-w1)//2, 0 ) )
    # im_new = np.asarray(im_new)  # convert "im_new" from "PIL Image" to "array"
    height, width = im.shape
    top = (32-height)//2
    bottom = 32-height-top
    left = (32-width)//2
    right = 32-width-left
    im_new = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0))
    # print(im_new.shape)
    return im_new

def findMin(arr):
    min = 99999
    id = -1
    for i, num in enumerate(arr):
        if num < min:
            min = num
            id = i
    return id, min

def isValidNum(first, second, neg, third, caresign = True):
    if first == '-' or first == 'empty':
        return False
    if second == '-' or second == 'empty':
        return False
    if third == '-' or third == 'empty':
        return False
    if neg != '-' and neg != 'empty' and caresign:
        return False
    return True

def temp2BitsValid(first, second, third):
    if first != "-" and first != "empty" and second != "-" and second != "empty" and third == "empty":
        return True
    else:
        return False

def insertText(img, text):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_PLAIN
    fontsize = 1
    color = (255,255,255)
    thickness = 1
    fonts, _ = cv2.getTextSize(text, font, fontsize, thickness) # 计算字体大小
    fontw, fonth = fonts
    loc = (int((w - fontw)/2), (h - fonth)) # 横向居中，竖向贴近底部
    cv2.putText(img, text, loc, font, fontsize, color, thickness)
    return img

def merge_image(image_list, x_num, y_num, height=None, width=None):
    # x_num: 行数
    # y_num: 列数
    new_image_list = []
    x_image_list = []
    for image in image_list:
        if (height is None) or (width is None):
            height, width = image.shape
        frame = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) # 每张图片的大小可自行resize
        new_image_list.append(frame)
    for x in range(x_num): # 形成一行n列图片
        htich = np.hstack([i for i in new_image_list[y_num*x:y_num*(x+1)]])
        x_image_list.append(htich)
    vtich = np.vstack(x_image_list) # 把行图片拼起来，形成m行n列图片
    return vtich

def drawImages_Gray(R_arr, Tm_arr, T_arr, I_arr, postfix):
    res = []
    if not R_arr is None:
        vtich = merge_image(R_arr, 1, 4)
        vh, vw = vtich.shape
        prompt_img = np.zeros((vh, vw), dtype=np.uint8)
        prompt_img = insertText(prompt_img, "R_"+postfix+":")
        res.append(merge_image([prompt_img, vtich], 2, 1))
    if not Tm_arr is None:
        vtich = merge_image(Tm_arr, 1, 4)
        vh, vw = vtich.shape
        prompt_img = np.zeros((vh, vw), dtype=np.uint8)
        prompt_img = insertText(prompt_img, "Tm_"+postfix+":")
        res.append(merge_image([prompt_img, vtich], 2, 1))
    if not T_arr is None:
        vtich = merge_image(T_arr, 1, 4)
        vh, vw = vtich.shape
        prompt_img = np.zeros((vh, vw), dtype=np.uint8)
        prompt_img = insertText(prompt_img, "T_"+postfix+":")
        res.append(merge_image([prompt_img, vtich], 2, 1))
    if not I_arr is None:
        vtich = merge_image(I_arr, 1, 4)
        vh, vw = vtich.shape
        prompt_img = np.zeros((vh, vw), dtype=np.uint8)
        prompt_img = insertText(prompt_img, "I_"+postfix+":")
        res.append(merge_image([prompt_img, vtich], 2, 1))    
    if len(res):
        merge = merge_image(res, len(res), 1)
        # print(res.shape)
        cv2.imshow("debug", merge)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyWindow("debug")
            exit(0)

def readTempate():
    tlist = []
    ned = cv2.imread(TEMPLATE_DIR+"E1.png")
    tlist.append(ned)
    ned = cv2.imread(TEMPLATE_DIR+"0.png")
    tlist.append(ned)
    ned = cv2.imread(TEMPLATE_DIR+"E2.png")
    tlist.append(ned)
    ned = cv2.imread(TEMPLATE_DIR+"E3.png")
    tlist.append(ned)
    return tlist

def obtainPerspective():
    p = cv2.imread(TEMPLATE_DIR+"perspective.png")
    p_gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(MAX_FEATURES)
    kp_ref, des_ref = sift.detectAndCompute(p_gray, None)
    return (sift, kp_ref, des_ref, p, p_gray)

def alignImages_SIFT(im, pers_tuple):
    sift, kp_ref, des_ref, p_color, p_gray = pers_tuple

    # Convert images to grayscale
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Detect SIFT features and compute descriptors.
    # len(keypoint) == len(descriptors) == MAX_FEATURES
    kp, des = sift.detectAndCompute(imGray, None)
    # Match features.
    # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    # matches = matcher.match(descriptors1, descriptors2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # sift的normType应该使用NORM_L2或者NORM_L1
    matches = list(bf.match(des, des_ref))
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    for m in matches:
        for n in matches:
            if(m != n and m.distance >= n.distance*0.75):
                matches.remove(m)
                break
    # Draw top matches
    # imMatches = cv2.drawMatches(im, kp, p_ref, kp_ref, matches, None)
    # cv2.imshow("matches", imMatches)
    # cv2.waitKey()
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kp[match.queryIdx].pt
        points2[i, :] = kp_ref[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width, channel = p_color.shape
    im1Reg = cv2.warpPerspective(im, h, (width, height))
    return im1Reg, h

def openVideo(videoname):
    video = cv2.VideoCapture(VIDEO_DIR + videoname)
    #check if the video exists
    if not video.isOpened():
        raise Exception("Video cannot be opened.")
    return video

def showVideoInfo(videoname, video, sampling_rate):
    #show some info of video
    fps = video.get(5)
    totalframes = video.get(7)
    print("Video: " + videoname)
    print("Total frames: " + str(totalframes))
    print("FPS: " + str(int(fps)))
    #if sampling_rate = 10, it means every 0.1s we capture a photo
    print("Sampling_rate: " + str(sampling_rate) + "Hz") 
    # timeF: we pick one frame every 'timeF' frames.
    # Here we pick one frame every 5 frames. 
    timeF = fps / sampling_rate 
    print("Frames needed to be extracted: " + str(int(totalframes/timeF)))
    return int(timeF), totalframes

def initExcelTable():
    #create and initialize an Excel table
    wb = xlwt.Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    sheet1.write(0, 0, "Time/s")
    sheet1.write(0, 1, "Resistance vacuum gauge/Pa")
    sheet1.write(0, 2, "Temperature/C")
    sheet1.write(0, 3, "Thermocouple vacuum gauge/Pa")
    sheet1.write(0, 4, "Ionization vacuum gauge/Pa")
    return wb, sheet1

def showProgress(cur, tot):
    print("\r", end="")
    print("progress: {:.1f}%".format(cur/tot*100), end="")
    sys.stdout.flush()

def saveResult(wb, videoname):  
    os.makedirs(RESULT_DIR, exist_ok=True)
    filename = videoname[:-4] +" "+str(datetime.datetime.now())[0:19].replace(":","-")+'.xls'
    wb.save(RESULT_DIR + filename)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "alignImages_SIFT")
        exit(1)
    elif sys.argv[1] != "alignImages_SIFT":
        print("Invalid param!")
        exit(1)

    # Read reference image
    p = cv2.imread("./36.png")
    p_gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(MAX_FEATURES)
    kp_ref, des_ref = sift.detectAndCompute(p_gray, None)
    pers_tuple = (sift, kp_ref, des_ref, p, p_gray)
    
    # Read image to be aligned
    imFilename = "1.png"
    print("Reading image to align: ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    
    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages_SIFT(im, pers_tuple)
    
    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)
    
    # Print estimated homography
    print("Estimated homography : \n",  h)