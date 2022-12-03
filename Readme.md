<div align="center">
<img src="./title.png" width="400px">

______________________________________________________________________

<p align="center">
  <a href="">AFM ICE Structure Prediction</a>
</p>


</div>

______________________________________________________________________

## Brief
This project was built to predict the AFM pictures of water, which is put into very low temperture.

## How To Use

### Step 0: Install

Make sure that the cuda is available for the computer doing training.

And using WSL2 or linux system to do it.

You can refer to the websides provided below.

  <a href="https://datawhalechina.github.io/dive-into-cv-pytorch/#/">A guide of pytorch in chinese</a>

  <a href="https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-o">Cuda installation</a>

  <a href="https://zhuanlan.zhihu.com/p/149848405">zhihu turtorial</a>

After installation, please check the command below to make sure whether it is correct.

```bash
nvcc -V
nvidia-smi
```

If you are using HPC, you can install the latest python via Miniconda.

```bash
# install miniconda
# get the latest conda on the website.
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装到自己的HOME目录下software/miniconda3中，这个目录在安装前不能存在；
sh Miniconda3-latest-Linux-x86_64.sh -b -p ${HOME}/software/miniconda3

# 安装成功后删除安装包
rm -f Miniconda3-latest-Linux-x86_64.sh

# 将环境变量写入~/.bashrc文件中；
echo "export PATH=${HOME}/software/miniconda3/bin:\$PATH" >> ~/.bashrc

# 退出重新登录或者执行以下命令
source ~/.bashrc
```

Also, it's highly recommended that create a virtual environment for this project.

Example of using virtualenv

```bash
# Creating enviroment
python3 -m venv [env]
# activate it
. ./[env]/bin/activate
```

Some python package are require in this project:

```bash
pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install jupyter tqdm opencv-python matplotlib numpy pandas yacs -i https://pypi.tuna.tsinghua.edu.cn/simple
```

After that,  you can try running the program to see if your computer prepare for it.

### Step 1: Using
Make sure that the dataset exists in the father dir. and data dir contains all of the training pictures. The file list has to be named as train_FileList, val_FileList and test_FileList.

-Dataset/

|

|-data/

|

|-train_FileList ...

There are several parsers can be used in the program example

```bash
python3 main.py --log-name balanced_local_2gpu.log --batch-size 32 --mode train --worker 12 --model ./model.pkl --dataset bulk_ice --local-epoch 0 --epoch 86 --gpu 0,1
# to see more use
python3 main.py --help
```

______________________________________________________________________

## 一些經驗
- worker就算在linux系統或是在超算上跑都不要設成0，超算上設n-nodes的數目;
- 如果出現了一些很tricky的bug，比如跑幾個epoch才報錯，建議花點時間把整個算法搬到CPU上，再debug，你真的會找不到到底哪出問題了(主要是顯卡上跑算法就算報錯print也會全跑光，而且會在同時輸出)
- 減少.cuda(),.cpu(),to()的使用，在創建tensor的時候直接在device上創建，真有數據傳輸的話，要加nonblocking = True。
- 避免使用太多for，能用張量計算就用張量計算，速度快很多，不過不是全部算法都能夠這樣做就是了，而且可讀性會有一定的減低。
- 不要嘗試將一切運算都設計成可反向傳播的，因為多出來的時間開消遠大於模型效能的改善
- 記住你是一個搞物理的人，不要花太多時間在搭系統搞封裝，因為科研代碼本身就需要瘋狂修改，而且後面接手你的工作的人能不能快速看懂又另一個問題了。
- 