<h1 style="font-size:50pt" align="center">Food Rating with Residual Network</h2>
<p align="center">Group 13, IS712, 2020 Fall</h2>

## Code Structure
```
is712/
├── dataset/
├── models/
├── const.py
├── dataloader.py
├── inference.py
├── main.py
├── models.py
└── utils.py
```


## Requirements

```
python 3.6
pytorch
torchvision
numpy
progressbar
```

## Running Our System
To run our rating system, simply following these steps in your Shell interactive interpreter:

```shell
>>> python3 main.py  # train the model and save parameters in '/PATH/TO/is712/models/'
>>> python3 inference.py  # do inference on test dataset
```
