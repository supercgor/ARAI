<div align="center">
<img src="./title.png" width="400px">

______________________________________________________________________

<p align="center">
  <a href="">Atomic Reconstruction on AFM Images</a>
</p>

</div>

______________________________________________________________________

## Brief
This project was built to predict the AFM pictures of water, which is put into very low temperture.

這項目是用於建立 AFM 圖像 對 水分子位置 的預測。

## Prerequisites
- Python 3 (tested with Python 3.10)
- PyTorch (tested with torch v2.0)
- Python packages as specified in [requirements.txt](requirements.txt)

## Installation
```
$ git clone https://github.com/supercgor/ARAI.git
$ cd ARAI
$ sudo pip3 install -r requirements.txt
```

## Prepare for datasets
The data should be placed in ARAI/datasets/data/EXAMPLE/ , In EXAMPLE, you should contain:
```
EXAMPLE/
  afm/
    name1/
      0.png
      1.png
      ...
  label/
    name1.poscar
    name2.poscar
    ...
  train.filelist
    name1
    name2
    ...
  valid.filelist
  test.filelist
```

## Prepare for the models
The model is placed in ARAI/model/MODELNAME, dir MODELNAME should cotain:
```
MODELNAME/
  runs(optional, tensorboard logdir)/
  train.log
  PKLNAME.pkl
  info.json(model constructing dict)
    "best": PKLNAME.pkl,
    ...
```

## Prepare for training
You have to set up the config.py according your computer condition. To start training:
```
$ python3 main.py --mode train
```

## Logging
You can check the train.log in model/MODELNAME/ . 

You can also use tensorboard to see more detail training log.
```
$ tensorboard --logdir model/MODELNAME/runs --port 6006
```
______________________________________________________________________

## Contributors
Tang Bin Ze, Qin Mian, Lo Chon Hei

This project is still working and there would be LOTS OF BUGS.