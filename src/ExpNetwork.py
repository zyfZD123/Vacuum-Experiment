import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

class MyNetwork(nn.Module):
    def __init__(self):
        self.output_size = 12 #12 types of images
        super(MyNetwork, self).__init__()
        #nn.Conv2d(ni,no,f,s) ni:number of input channels, no: num of output channels
        #f: convolutional kernel size, usually 5. s:stride (default=1), if it's not 1, it will not
        #use the whole data
        self.conv1 = nn.Conv2d(1, 6, 5) # input: one channel, because it is in GRAY scale
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(400, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, self.output_size)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MyDataset_notfromdisk(Dataset): 
    # default mode is "createdataset", you may set it to "test" if necessary
    def __init__(self, imglist, mode, transform = None):
        imgs = []
        if mode=="test":
            for img1, img2, neg, img3 in imglist:
                imgs.append((img1, 1)) # all tag of images is 1 because we don't use tag
                imgs.append((img2, 1))
                imgs.append((neg, 1))
                imgs.append((img3, 1))
        elif mode=="createdataset":
            for onetuple in imglist:
                imgs.append(onetuple)
        else:
            raise Exception("MyDataset_notfromdisk: unknown mode.")
        self.imgs = imgs 
        self.transform = transform
        
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.fromarray(fn)
        if self.transform is not None:
            img = self.transform(img) 
        return img, label
        
    def __len__(self):
        return len(self.imgs)