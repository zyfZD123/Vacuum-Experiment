## Readme

### How to run

#### Environment (For reference only)

- Python version: 3.10.5
- OpenCV version: 4.6.0.66
- Torch version: 1.12.1+cpu
- Device information: Windows 10 or Windows 11, 64-bit, 16GB RAM, intel i7-core


#### If reading experiment data is needed

- Put the network model("resultv3_gray.pth" for example) in the root directory
- Put the video in "videos" directory
- Turn on a terminal in the root directory, input:


```shell
    cd src
    py ReadExpData.py
```

- Wait for the result in "resultData" directory.


#### If training a new network is needed

- Put the training images in "trainset"
- Turn on a terminal in the root directory, input:


```shell
    cd src
    py TrainNetwork.py

```

- Wait for the result in the root directory.


### Introduction

#### File: ReadExpData.ipynb

- **If you already have a .pth file of a CNN and some videos, just use this file.**
- Check the file for more details.

#### File: TrainNetwork.ipynb

- Use it to train a CNN.
- Check the file for more details.

#### File: GenerateDataset.ipynb

- This file is used to generate new dataset based on your old CNN.
- Check the file for more details.

#### File: ExpNetwork.py

- Contain the definition of CNN and Dataset. 
- Indirectly used in other modules, such as ReadExpData.ipynb and TrainNetwork.ipynb.

#### File: resultv3_gray.pth

- CNN parameters file.
- Indirectly used in ReadExpData.ipynb.

### Debug log

#### Finished

- In `ReadExpData.ipynb`: identify negative number (negative exponent)
- In `ReadExpData.ipynb`: multi-gauge identification

#### Developping

- In `ReadExpData.ipynb`: need to cut images more flexibly...
- In `GenerateDataset.ipynb`: need to arrange the code to generate all types of datasets(different threshold, different transform, etc)...