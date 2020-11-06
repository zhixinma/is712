<h1 style="font-size:50pt" align="center">Food Rating with Residual Network</h2>
<p align="center">Group 13, IS712, 2020 Fall</h2>


## Requirements

The Following requirements should be installed in python3.6 environment.
```
python3.6
pytorch
torchvision
numpy
progressbar
```

## Running Our System

### Code Structure
```
is712/
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
├── model/
│   ├── best_cnn_lrs_0
│   ├── best_resnet_cnn_lrs_0
│   ├── best_resnet_lrs_0
│   ├── best_resnet_lrs_dis_reg_0
│   └── best_resnet_lrs_val_0
├── const.py
├── main.py
├── inference.py
├── dataloader.py
├── models.py
└── utils.py
```
1. Raw data (e.g. ```train/```, ```validation/```, ```test/```) should be extracted to ```/PATH/TO/is712/dataset/```.
2. Checkpoints of different model variants (used for phase II) are saved in ```/PATH/TO/is712/model/```.
3. ```const.py```: Configuraion variables.
4. ```main.py```: Training script.
5. ```inferemce.py```: Inference script.
6. ```dataloader.py```: I/O classes and functions.
7. ```models.py```: Model variants.
8. ```utils.py```: Utility functions.

### Get Start
To run our rating system, simply following these steps in your Shell interactive interpreter:
```shell
>>> python3 main.py  # train the model and save parameters in '/PATH/TO/is712/models/'
>>> python3 inference.py  # do inference on test dataset
```

You may change the variables in ```const.py``` for differnet settings:
```python
TOY = False  # test code with toy data
SEPARATE = False  # separate parameters of mean and variance prediction
USE_FULL_DATA = False  # use all training data
MEAN_MODEL_NAME = "resnet"  # name of mean prediction model
VAR_MODEL_NAME = ""  # name of mean prediction model
DISCRETE_REG = False  # use regression and map result into discrete_means/discrete_vars
GPU_NUM = 0  # index of the gpu unit to use
PREFIX = "resnet_lrs_val"  # model prefix
```
The final result is the average prediction of the five models, plsease refer to the report for model details. \
To run different model variants, you can just replace these variables (on the top of ```const.py```) with the following settings. \
Note that ```GPU_NUM``` should be replaced with the available one according to your situation.

**CNN**
```python
TOY = False  # test code with toy data
SEPARATE = False  # separate parameters of mean and variance prediction
USE_FULL_DATA = True  # use all training data
MEAN_MODEL_NAME = "resnet"  # name of mean prediction model
VAR_MODEL_NAME = ""  # name of mean prediction model
DISCRETE_REG = False  # use regression and map result into discrete_means/discrete_vars
GPU_NUM = 0  # index of the gpu unit to use
PREFIX = "cnn_lrs"  # model prefix
```

**ResNet**
```python
TOY = False  # test code with toy data
SEPARATE = False  # separate parameters of mean and variance prediction
USE_FULL_DATA = True  # use all training data
MEAN_MODEL_NAME = "cnn"  # name of mean prediction model
VAR_MODEL_NAME = ""  # name of mean prediction model
DISCRETE_REG = False  # use regression and map result into discrete_means/discrete_vars
GPU_NUM = 0  # index of the gpu unit to use
PREFIX = "resnet_lrs"  # model prefix
```

**ResNet+CNN**
```python
TOY = False  # test code with toy data
SEPARATE = True  # separate parameters of mean and variance prediction
USE_FULL_DATA = True  # use all training data
MEAN_MODEL_NAME = "resnet"  # name of mean prediction model
VAR_MODEL_NAME = "cnn"  # name of mean prediction model
DISCRETE_REG = False  # use regression and map result into discrete_means/discrete_vars
GPU_NUM = 0  # index of the gpu unit to use
PREFIX = "resnet_cnn_lrs"  # model prefix
```

**Discrete ResNet**
```python
TOY = False  # test code with toy data
SEPARATE = False  # separate parameters of mean and variance prediction
USE_FULL_DATA = True  # use all training data
MEAN_MODEL_NAME = "resnet"  # name of mean prediction model
VAR_MODEL_NAME = ""  # name of mean prediction model
DISCRETE_REG = True  # use regression and map result into discrete_means/discrete_vars
GPU_NUM = 0  # index of the gpu unit to use
PREFIX = "resnet_lrs_dis_reg"  # model prefix
```

**Cross Validation**
```python
TOY = False  # test code with toy data
SEPARATE = False  # separate parameters of mean and variance prediction
USE_FULL_DATA = False  # use all training data
MEAN_MODEL_NAME = "resnet"  # name of mean prediction model
VAR_MODEL_NAME = ""  # name of mean prediction model
DISCRETE_REG = False  # use regression and map result into discrete_means/discrete_vars
GPU_NUM = 0  # index of the gpu unit to use
PREFIX = "resnet_lrs_val"  # model prefix
```

