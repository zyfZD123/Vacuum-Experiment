## Readme

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