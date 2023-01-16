"""
#### File: TrainNetwork.ipynb
- This file will train a convolutional neural network according to the trainset
- The main function `trainNetwork()` is at the bottom. 
  **Please ensure that all of code above `trainNetwork()` 
  has been compiled and run before launching it.**
- Just compile and run all the code sequentially.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from ExpNetwork import *
from torch.utils.data import DataLoader

"""
#### functions in `trainNetwork()`
- `getTrainLoader()`: using ImageFolder and Dataloader to get a training dataloader.
- `printTrainResult()`: print loss and CNN parameters.
"""

def getTrainLoader(batchsize, trainsetdir):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5 ), (0.5)),
        transforms.Grayscale(num_output_channels=1) #convert to grayscale
        ])

    train = ImageFolder(root=trainsetdir, transform=train_transform)
    trainloader = DataLoader(train, batch_size = batchsize, shuffle = True)
    # the second return value is number of images in trainset
    return trainloader, len(train)

def printTrainResult(epochs, net, losslist, lossfnresult, shownetparam):
    print('Finished training.')
    if lossfnresult == True:
        plt.figure(num=1, figsize=(10,8), facecolor="white", edgecolor="white")
        plt.plot(losslist)
        plt.title("Loss change")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xticks(np.arange(0, epochs, 5))
        plt.yticks(np.arange(0, losslist[0], 100))
        plt.show()
        print("Loss:", losslist)
    if shownetparam == True:
        print('Net parameters:')
        for key, value in net.state_dict().items():
            torch.set_printoptions(profile="full")
            print(str(key)+": ", value)

def trainNetwork(   networkname, batchsize, epochs, trainsetdir, learning_rate = 1e-2, 
                    lossfnresult = True, consoledebug = True, shownetparam = False):
    # Check ExpNetwork.py for details of MyNetwork
    net = MyNetwork()
    #note: no need to do softmax, because softmax is done in function "nn.CrossEntropyLoss()"
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = learning_rate)
    trainloader, _ = getTrainLoader(batchsize=batchsize, trainsetdir=trainsetdir)
    losslist = []
    print("Start training...")
    for epoch in range(epochs):
        loss1 = 0
        for mydata in trainloader:
            inputs, labels = mydata
            optimizer.zero_grad()
            predicted_labels = net(inputs)
            loss = loss_function(predicted_labels, labels)
            loss.backward()
            optimizer.step()
            loss1 += loss.item()
        if consoledebug == True:
            losslist.append(loss1)
            print('Epoch: %d, loss: %.6f' % (epoch + 1, loss1))
    # save parameters of CNN to current directory    
    torch.save(net.state_dict(), '../' + networkname)   
    printTrainResult(epochs, net, losslist, lossfnresult, shownetparam)
    

"""
#### Main function: trainNetwork()
- parameters:
  - networkname: the name of file that stores all parameters of network. 
    **<font color="red">Note that it will overwrite the file with same name in 
    current directory! So please ensure the networkname is unique</font>**
  - trainsetdir: you may set it to your own dataset directory if neccessary.
  - batchsize: for network training
  - epoch: for network training
  - learning rate(optional, default is 0.01)
  - lossfnresult(optional, default is True): to check the loss function's value, set it True
  - consoledebug(optional, default is True): to trace the current epoch, set it True, but it may print lots of lines
  - shownetparam(optional, default is False): to check all the parameters of CNN
"""

if __name__ == "__main__":
    trainNetwork(   networkname = "resultv3_justfortest.pth", 
                    batchsize=30, 
                    epochs=30,  
                    trainsetdir="../trainsetv4", 
                    lossfnresult=True, 
                    shownetparam=True, 
                    consoledebug=True   )