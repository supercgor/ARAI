<div align="center">
<img src="./demo/title.png" width="400px">

______________________________________________________________________

<p align="center">
  <a href="">Atomic Reconstruction on AFM Images</a>
</p>

</div>

______________________________________________________________________

## Brief
This project was built to predict the AFM pictures of water, which is put into very low temperture.

这项目是用于建立 AFM 图像 对 水分子位置 的预测。

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
The path of data should be formatted as follow:
```
data/
  dataset1/
    *.filelist
    afm/
      imgs1/
        0.png
        1.png
        ...
      imgs2/
        0.png
        1.png
        ...
      ...
    label/
      imgs1.poscar
      imgs2.poscar
      ...
    datapack/ (optional)
      imgs1.npz
      imgs2.npz
      ...
    result/ (optional)
      modelname/
        imgs1.poscar
        imgs2.poscar
      ...
```

## Prepare for the models
The model is placed in ARAI/model/pretrain/modelname, dir modelname should contain:
```
modelname/
  a.pkl
  b.pkl
  *.log (optional)
  runs/ (optional)
```

## Prepare for training
You have to set up the config.py & wm.py according your computer condition. There are the codes you may need to change:
```python
# checkpoint name
_C.model.checkpoint = "the model dir name"
# use net
_C.model.fea = "the model name in that dir"
_C.model.reg = "the model name in that dir"
_C.model.cyc = "the model name in that dir"
```
To start training:
```
$ python3 train.py
$ python3 cycTrain.py
$ python3 cycTune.py
$ python3 test.py --dataset ../data/dataset1 --filelist test.filelist --label False --npy False
```

## Logging
You can check the train.log in model/modelname/ . 

You can also use tensorboard to see more detail training log.
```
$ tensorboard --logdir model/modelname/runs --port 6006
```
______________________________________________________________________

## Contributors
Tang Bin Ze, Qin Mian, Lo Chon Hei

This project is still working and there could be LOTS OF BUGS.