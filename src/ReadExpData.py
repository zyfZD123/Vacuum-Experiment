"""
#### File: ReadExpData.ipynb
- This file will read all video you provide, extract experiment data, 
  and put it into corresponding excel. **Please check results in directory "resultData".**
- The main function `readExpData()` is at the bottom. 
  **Please ensure that all of code above `readExpData()` has been compiled 
  and run before launching the main function.**
- Just compile and run in the order of the code.
- To see the introduction of all functions, you'd better see them from bottom to top.
"""

"""
##### some imports and global variables
- You may need to install opencv-python, matplotlib, torch, numpy, torchvision, xlwt, Pillow.
- If you run it in VsCode, you may add the path of "ExpNetwork.py" so that it can run smoothly. 
  Please check "settings: Python > Analysis: Extra Paths" and add the path to it.
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import ExpNetwork
import Utils

custom_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5 ), (0.5))],
    )
classes = ('-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'empty')

torch.set_printoptions(profile="full")

"""
#### assisting functions
- `processImage()`:
  - It will be called in `splitSave()`. Initially it gets a image, a template image, a type.
  - Then it call `cv2.matchTemplate()` to locate the number. 
  	For example, given an image of template 'E', it locates 'E'(see the variable `max_loc`) 
	and other numbers according to the relative position of 'E'.
  - Different types mean different guages to be read. For each guage, we set the relative 
  	position by hand. You may adjust the relative position by hand if necessary.
  - This function will also pre-process the image, such as converting from BGR to GRAY, 
  	thresholding, resizing, and dilating.
- `netRead()`: will be called in `readNumber()` to read number from a batch provided by dataloader.
"""

def processImage(img, ned, gaugetype):
    result = cv2.matchTemplate(img, ned, cv2.TM_CCOEFF_NORMED)
    #Note: img and ned are both in BGR, not in RGB!
    if gaugetype == "R":
        # original size of sample1, sample2, sample3: 26*26
        # for example, if img.shape= (360, 426, 3), ned.shape= (26, 21, 3), 
        # then result.shape= (360-(26-1)=335, 426-(21-1)=406)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, test_img = cv2.threshold(test_img, 150, 255, cv2.THRESH_BINARY)
        # 21*21
        sample1 = test_img[max_loc[1] : max_loc[1]+26, max_loc[0]-50 : max_loc[0]-24] 
        sample2 = test_img[max_loc[1] : max_loc[1]+26, max_loc[0]-25 : max_loc[0]+1]
        negsample = test_img[max_loc[1] : max_loc[1]+26, max_loc[0]+23 : max_loc[0]+49]
        sample3 = test_img[max_loc[1]-1 : max_loc[1]+25, max_loc[0]+48 : max_loc[0]+74]
    elif gaugetype == "Tm":
        # 多目标匹配取最左边那个0
        index = np.where(result > 0.8)
        if len(index[1]) > 0:
            yPos, _ = Utils.findMin(index[1])
            max_loc = (index[1][yPos], index[0][yPos])
        else:
            _, _, _, max_loc = cv2.minMaxLoc(result)
        test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, test_img = cv2.threshold(test_img, 100, 255, cv2.THRESH_BINARY)
        
        # 识别0标志的参数, 32*32
        sample1 = test_img[max_loc[1]+4 : max_loc[1]+36, max_loc[0]+27 : max_loc[0]+59] 
        sample2 = test_img[max_loc[1]+4 : max_loc[1]+36, max_loc[0]+53 : max_loc[0]+85]
        negsample = test_img[0:32, 0:32] #useless, since temperature cannot be negative
        sample3 = test_img[max_loc[1]+4 : max_loc[1]+36, max_loc[0]+81 : max_loc[0]+113] 
        
        # 识别oC标志的参数，作为备用：32*23
        # sample1 = test_img[max_loc[1]+8 : max_loc[1]+40, max_loc[0]-87 : max_loc[0]-64] #之前是+9，+42，-84，-62
        # sample2 = test_img[max_loc[1]+8 : max_loc[1]+40, max_loc[0]-60 : max_loc[0]-37] # 之前是+7，+40，-57，-34
        # negsample = test_img[0:32, 0:32] #useless, since temperature cannot be negative
        # sample3 = test_img[max_loc[1]+8 : max_loc[1]+40, max_loc[0]-30 : max_loc[0]-7] #之前是+8，+41，-28，-5
    elif gaugetype == "T":
        _, _, _, max_loc = cv2.minMaxLoc(result)
        test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, test_img = cv2.threshold(test_img, 150, 255, cv2.THRESH_BINARY)
        # 27*27
        sample1 = test_img[max_loc[1]+2 : max_loc[1]+32, max_loc[0]-74 : max_loc[0]-44]
        sample2 = test_img[max_loc[1]+2 : max_loc[1]+32, max_loc[0]-38 : max_loc[0]-8]
        negsample = test_img[max_loc[1] : max_loc[1]+30, max_loc[0]+33 : max_loc[0]+63]
        sample3 = test_img[max_loc[1] : max_loc[1]+30, max_loc[0]+69 : max_loc[0]+99]
    elif gaugetype == "I":
        _, _, _, max_loc = cv2.minMaxLoc(result)
        test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, test_img = cv2.threshold(test_img, 150, 255, cv2.THRESH_BINARY)
        # 30*30
        sample1 = test_img[max_loc[1]+4 : max_loc[1]+34, max_loc[0]-74 : max_loc[0]-44] 
        sample2 = test_img[max_loc[1]+3 : max_loc[1]+33, max_loc[0]-38 : max_loc[0]-8] 
        negsample = test_img[max_loc[1]+3 : max_loc[1]+33, max_loc[0]+40 : max_loc[0]+70]
        sample3 = test_img[max_loc[1] : max_loc[1]+30, max_loc[0]+75 : max_loc[0]+105]
    
    sample1 = Utils.pad_image(sample1, 32, 32)
    sample2 = Utils.pad_image(sample2, 32, 32)
    sample3 = Utils.pad_image(sample3, 32, 32)
    if gaugetype != "Tm":
        negsample = Utils.pad_image(negsample, 32, 32)
    if gaugetype == "Tm":
        kernel = np.ones((3,3), np.uint8)
    else:
        kernel = np.ones((1,1), np.uint8)
    sample1 = cv2.dilate(sample1, kernel)
    sample2 = cv2.dilate(sample2, kernel)
    sample3 = cv2.dilate(sample3, kernel)
    return sample1, sample2, negsample, sample3

def netRead(batch, net):
    images = batch[0]
    outputs = net(images)
    sm = nn.Softmax(dim=1)      
    sm_outputs = sm(outputs)
    probs, index = torch.max(sm_outputs, dim=1)
    first = classes[index[0]]
    second = classes[index[1]]
    neg = classes[index[2]]
    third = classes[index[3]]
    return first, second, neg, third

"""
#### functions in `ReadExpData()`
- `loadNet()`: load parameters of a CNN from the current directory
- `splitSave()`: 
  - According to the given ratio from @param "splitarray" in `readExpData()`, 
    cut the frame of video into four parts. For example, if the third tuple of 
    "splitarray" is (0, 0.5, 0.4, 0.8), and the size of frame is 1280*720, 
    `splitSave()` will cut the guage "T" of the frame, 
    which is `frame[1280*0:1280*0.5][720*0.4:720*0.8]`
  - Then it passes the split frame to `processImage()`, which is introduced above.
- `readNumber()`: it will use the CNN to identify the number, 
  calculate the result and write it to the workbook.
"""

def splitSave(frame, splitarray, templateimglist, debugmode = []):
    width, length, _ = np.shape(frame)
    # Note：Resistance vacuum gauge--R， Thermocouple vacuum gauge--T， Ionization vacuum gauge--I
    #       temperature--Tm
    # Resistance vacuum gauge
    widthlower, widthupper, lengthlower, lengthupper = splitarray[0]
    subframe = frame[int(widthlower*width):int(widthupper*width), int(lengthlower*length):int(lengthupper*length)]
    first_R, second_R, neg_R, third_R = processImage(subframe,templateimglist[0],"R")
    # temperature
    widthlower, widthupper, lengthlower, lengthupper = splitarray[1]
    subframe = frame[int(widthlower*width):int(widthupper*width), int(lengthlower*length):int(lengthupper*length)]
    first_Tm, second_Tm, neg_Tm, third_Tm = processImage(subframe,templateimglist[1],"Tm")
    # # Thermocouple vacuum gauge
    widthlower, widthupper, lengthlower, lengthupper = splitarray[2]
    subframe = frame[int(widthlower*width):int(widthupper*width), int(lengthlower*length):int(lengthupper*length)]
    first_T, second_T, neg_T, third_T = processImage(subframe,templateimglist[2],"T")
    # Ionization vacuum gauge
    widthlower, widthupper, lengthlower, lengthupper = splitarray[3]
    subframe = frame[int(widthlower*width):int(widthupper*width), int(lengthlower*length):int(lengthupper*length)]
    first_I, second_I, neg_I, third_I = processImage(subframe,templateimglist[3],"I")
    
    R_arr = None
    Tm_arr = None
    T_arr = None
    I_arr = None
    debugflag = False

    if 'R' in debugmode:
        R_arr = [first_R, second_R, neg_R, third_R]
        debugflag = True
    if 'Tm' in debugmode:
        Tm_arr = [np.zeros((first_Tm.shape), dtype=np.uint8), first_Tm, second_Tm, third_Tm]
        debugflag = True
    if 'T' in debugmode:
        T_arr = [first_T, second_T, neg_T, third_T]
        debugflag = True
    if 'I' in debugmode:
        I_arr = [first_I, second_I, neg_I, third_I]
        debugflag = True
    if debugflag:
        Utils.drawImages_Gray(R_arr, Tm_arr, T_arr, I_arr, "Split")
    return (first_R, second_R, neg_R, third_R, 
            first_Tm, second_Tm, neg_Tm, third_Tm, 
            first_T, second_T, neg_T, third_T, 
            first_I, second_I, neg_I, third_I)

def readNumber(net, sheet, debugmode = [], show_result = False,  imageslist_R = None, 
                imageslist_Tm = None, imageslist_T = None, imageslist_I = None):#test three images
    if imageslist_R is None or imageslist_Tm is None or imageslist_T is None or imageslist_I is None:
        raise Exception('At least one imageslist not found.')
    
    # R
    real_test_R = ExpNetwork.MyDataset_notfromdisk(imglist=imageslist_R, transform=custom_transform, mode="test")
    real_testloader_R = DataLoader(real_test_R, batch_size = 4, shuffle = False)
    for i, batch in enumerate(real_testloader_R, 1):
        # structure of batch: 
        # len(batch) == 2
        # batch[0] is a list of images, batch[1] is a list of labels
        # len(batch[0]) == len(batch[1]) == batch_size
        # label of batch[0][i] is batch[1][i]
        first, second, neg, third = netRead(batch, net)
        if 'R' in debugmode:
            Utils.drawImages_Gray([np.array(batch[0][0][0]), np.array(batch[0][1][0]), 
                                np.array(batch[0][2][0]), np.array(batch[0][3][0])],
                None, None, None, "recognize"                
            )
            print("R: first={}, second={}, neg={}, third={}".format(first, second, neg, third))
        if Utils.isValidNum(first, second, neg, third):
            if neg == 'empty':
                result = (eval(first) + eval(second)*0.1)*(10**eval(third))
            elif neg == '-':
                result = (eval(first) + eval(second)*0.1)*(10**(-eval(third)))
        else:
            result = "NaN"
        sheet.write(i, 1, result)
        sheet.write(i, 0, i/10) #'result_cnt/10' means time(units of sec)
        if show_result:
            print(result, i/10) #for debug
    
    # Tm
    real_test_Tm = ExpNetwork.MyDataset_notfromdisk(imglist=imageslist_Tm, transform=custom_transform, mode="test")
    real_testloader_Tm = DataLoader(real_test_Tm, batch_size = 4, shuffle = False)
    for i, batch in enumerate(real_testloader_Tm, 1):
        first, second, neg, third = netRead(batch, net)
        if 'Tm' in debugmode:
            h, w = batch[0][0].permute(1, 2, 0).shape
            Utils.drawImages_Gray(  None, [np.zeros((h, w), dtype=np.uint8), batch[0][0], 
                                    batch[0][1].permute(1, 2, 0), batch[0][3].permute(1, 2, 0)],
                                    None, None, "recognize"
            )
            print("Tm: first={}, second={}, third={}".format(first, second, third))
        if Utils.isValidNum(first, second, neg, third, False):
            result = eval(first)*100 + eval(second)*10 + eval(third)
        elif Utils.temp2BitsValid(first, second, third):
            result = eval(first)*10 + eval(second)
        else:
            result = "NaN"
        sheet.write(i, 2, result)

    # T
    real_test_T = ExpNetwork.MyDataset_notfromdisk(imglist=imageslist_T, transform=custom_transform, mode="test")
    real_testloader_T = DataLoader(real_test_T, batch_size = 4, shuffle = False)
    for i, batch in enumerate(real_testloader_T, 1):
        first, second, neg, third = netRead(batch, net)
        if 'T' in debugmode:
            Utils.drawImages_Gray(None, None, [batch[0][0].permute(1, 2, 0), batch[0][1].permute(1, 2, 0), 
                                batch[0][2].permute(1, 2, 0), batch[0][3].permute(1, 2, 0)],
                                None, "recognize"
            )
            print("T: first={}, second={}, neg={}, third={}".format(first, second, neg, third))
        if Utils.isValidNum(first, second, neg, third):
            if neg == 'empty':
                result = (eval(first) + eval(second)*0.1)*(10**eval(third))
            elif neg == '-':
                result = (eval(first) + eval(second)*0.1)*(10**(-eval(third)))
        else:
            result = "NaN"
        sheet.write(i, 3, result)

    # I
    real_test_I = ExpNetwork.MyDataset_notfromdisk(imglist=imageslist_I, transform=custom_transform, mode="test")
    real_testloader_I = DataLoader(real_test_I, batch_size = 4, shuffle = False)
    for i, batch in enumerate(real_testloader_I, 1):
        first, second, neg, third = netRead(batch, net)
        if 'I' in debugmode: 
            Utils.drawImages_Gray(  None, None, None, [batch[0][0].permute(1, 2, 0), 
                                    batch[0][1].permute(1, 2, 0), batch[0][2].permute(1, 2, 0), 
                                    batch[0][3].permute(1, 2, 0)], "recognize"
            )
            print("I: first={}, second={}, neg={}, third={}".format(first, second, neg, third))
        if Utils.isValidNum(first, second, neg, third):
            if neg == 'empty':
                result = (eval(first) + eval(second)*0.1)*(10**eval(third))
            elif neg == '-':
                result = (eval(first) + eval(second)*0.1)*(10**(-eval(third)))
        else:
            result = "NaN"
        sheet.write(i, 4, result)

def loadNet(netpath):
    net = ExpNetwork.MyNetwork()
    net.load_state_dict(torch.load(netpath))
    return net

def readExpData(videonamelist, netpath, splitarray, sampling_rate = 10, savetodisk = False, 
                fnReadNumDebug = [], fnSplitDebug = []):
    # load the trained convolution network
    net = loadNet(netpath)
    # load the template image for cv2.matchTemplate
    templateimglist = Utils.readTempate()
    # load the information of front perspective
    pers_tuple = Utils.obtainPerspective()
    # pers_im = cv2.imread(Utils.TEMPLATE_DIR + "perspective.png")

    # Read the video
    for videoname in videonamelist:
        video = Utils.openVideo(videoname)
        # timeF: we pick one frame every 'timeF' frames.
        # Here we pick one frame every 5 frames. 
        timeF, totalframes = Utils.showVideoInfo(videoname, video, sampling_rate)
        wb, sheet1 = Utils.initExcelTable()
        rval = True
        frame_cnt = 1
        imageslist_R = []
        imageslist_Tm = []
        imageslist_T = []
        imageslist_I = []
        cnt=0
        print("Splitting the image...")
        while rval: 
            # Keep reading frames until rval=False(that is, end of file)
            rval, frame = video.read() # Note: frame is in BGR colorspace, not RGB!
            if (frame_cnt % timeF == 0 and rval): 
                frame, h = Utils.alignImages_SIFT(frame, pers_tuple)
                # take down the data
                (first_R, second_R, neg_R, third_R, 
                 first_Tm, second_Tm, neg_Tm, third_Tm, 
                 first_T, second_T, neg_T, third_T, 
                 first_I, second_I, neg_I, third_I
                ) = splitSave(frame, splitarray, templateimglist, fnSplitDebug)
                imageslist_R.append((first_R, second_R, neg_R, third_R))
                imageslist_Tm.append((first_Tm, second_Tm, neg_Tm, third_Tm))
                imageslist_T.append((first_T, second_T, neg_T, third_T))
                imageslist_I.append((first_I, second_I, neg_I, third_I))
                # cnt+=1
            frame_cnt += 1
            if frame_cnt % 50 == 0:
                Utils.showProgress(frame_cnt, totalframes)
            # if cnt == 50:
            #     break
        #save the excel table
        print("\nReading the number...    ", end="")
        readNumber(net=net, sheet=sheet1, imageslist_R=imageslist_R, 
                    imageslist_Tm=imageslist_Tm, imageslist_T=imageslist_T, imageslist_I=imageslist_I,
                    debugmode=fnReadNumDebug)
        Utils.saveResult(wb, videoname)
        print("Done!\n")


"""
#### Main function
- parameters:
  - videonamelist: a list contains all names of video you want to extract experiment data. 
    Note that all video **should be at the directory "videos".** 
    For example, `videonamelist = ["00008.mp4", "00009.MTS"]`.
  - netpath: the name of file of the network you previously trained. 
    For example, `netpath="../resultv3_gray.pth"`
  - fnReadNumDebug: an array for debugging. For instance, `fnReadNumDebug=['Tm', 'I']` 
    means you want to check the results in function `readNumber()`
  - fnSplitDebug: an array for debugging of function `splitSave()`, like `fnReadNumDebug`.
  - splitarray: the ratio of how to cut the images. Check the details in introduction above: 
    **functions in `ReadExpData()`**
  - sampling_rate: its unit is Hz. `sampling_rate=10` means we snapshot every 0.1 seconds.
"""

if __name__ == "__main__":
    #Main function: readExpData()
    #for each video, you may change the variable 'videoname'
    readExpData(    videonamelist = ["手持相机.mp4", "00023_Trim.mp4"],
                    sampling_rate = 10,
                    netpath=Utils.MODEL_DIR + "resultv3_gray.pth",
                    fnSplitDebug=[], 
                    fnReadNumDebug=[], 
                    splitarray=[(0, 1/2, 0, 1/3), (0.4, 0.99999, 0, 0.5), (0, 0.5, 0.3, 0.67), (0, 0.5, 0.6, 0.99999)])