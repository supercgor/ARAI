import os, json, torch, cv2, time, logging, logging.handlers
import numpy as np
from torch import nn
from collections import OrderedDict
from network.unet3d_model import UNet3D as model

def Loader(cfg):
    name = cfg.path.checkpoint
    config = {"best_model": "", 
              "network": "unet", 
              "Z": cfg.DATA.Z,
              "dataset": cfg.path.dataset,
              "channel": cfg.MODEL.CHANNELS, 
              "other": time.asctime(time.localtime(time.time()))}
    new = False
    if  name == "None" or not os.path.exists(f"{cfg.path.check_root}/{name}"):
        new = True

    name = f"{config['network'].lower()}-{time.strftime('%m%d-%H:%M', time.localtime())}"
    os.mkdir(f"{cfg.path.check_root}/{name}")
    
    model_root = f"{cfg.path.check_root}/{name}"
    logger = Logger(model_root, elem = cfg.DATA.ELE_NAME, split = cfg.OTHER.SPLIT)
    ml = modelLoader(name = name, path = model_root, keeps = 5)
    if new:
        logger.info(f"No model is loaded! Start training a new one")
        ml.new(config)
    else:
        info = ml.load(f"{cfg.path.check_root}/{cfg.path.checkpoint}")
        logger.info(info)
    
    if len(cfg.TRAIN.DEVICE) > 1:
        parallel = True
    else:
        parallel = False

    #　ml.cuda(parallel = parallel)

    return ml, ml.model, logger


class modelLoader():
    def __init__(self, name = "None", path = "/home/supercgor/gitfile/data/model", keeps = 5):
        self.name = name
        self.root = path
        self.keeps = keeps
        self.keeps_name = []
        self.parallel = False
        # check
    
    def new(self, config):
        self._model = model(1, channels = config['channel'], output_z = config['Z'])
        self._model.weight_init()
        self.config = config
        return 

    def load(self, path):
        # load old model
        with open(f"{path}/info.json") as f:
            self.config = json.load(f)
        
        if self.config['network'] == "unet":        
            self._model = model(1, self.config['channel'], self.config['Z'])
        
        # try:
        self._model.load_state_dict(torch.load(f"{path}/{self.config['best_model']}"))
        info = f"Load model parameters from {path}/{self.config['best_model']}"
        # except RuntimeError:
        #     match_list = model.load_pretrained_layer(f"{path}/{self.config['best_model']}")
        #     info = f"Different model! Load match layers: {match_list}"

        return info

    def cuda(self, parallel=False):
        self.parallel = parallel
        self._model = self._model.cuda()
        if parallel:
            device_ids = list(map(int,os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            self._model = nn.DataParallel(self._model, device_ids=device_ids)
    
    @property
    def model(self):
        return self._model
    
    def save(self, name):
        if len(self.keeps_name) == self.keeps:
            # model save num == threshold, delete the oldest one and save new model
            # delete the oldest model
            os.remove(self.keeps_name.pop(0))

        # save new model
        state_dict = self._model.state_dict()
        if self.parallel:
            state_dict = self._Parallel2Single(state_dict)
        torch.save(state_dict, f"{self.root}/{name}")
        self.config['best_model'] = name
        self.keeps_name.append(f"{self.root}/{name}")
        info = open(f"{self.root}/info.json","w")
        json.dump(self.config, info)
        info.close()
    
    @staticmethod
    def _Parallel2Single(state_dict):
        """
        将并行的权值参数转换为串行的权值参数
        :param state_dict : 原始串行权值参数
        :return             : 并行的权值参数
        """
        converted = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            converted[name] = v

        return converted
    
class sampler():
    def __init__(self, name, path ="/home/supercgor/gitfile/data"):
        self.abs_path = f"{path}/{name}"
        if not os.path.exists(self.abs_path):
            raise FileNotFoundError(f"Not such dataset in {self.abs_path}")
        self.datalist = os.listdir(f"{self.abs_path}/afm")
    
    def __getitem__(self, index):
        img_path = f"{self.abs_path}/afm/{self.datalist[index]}"
        pl = poscarLoader(f"{self.abs_path}/label")
        info, positions = pl.load(f"{self.datalist[index]}.poscar")
        images = []
        for path in sorted(os.listdir(img_path), key=lambda x: int(x.split(".")[0])):
            img = cv2.imread(f"{img_path}/{path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
        
        return {"name": self.datalist[index],"info": info, "image": images, "position": positions}
    
    def get(self, name):
        index = self.datalist.index(name)
        return self.__getitem__(index)
    
    def get_npy(self, index):
        loc = f"{self.abs_path}/npy/{self.datalist[index]}.npy"
        pred = np.load(loc)
        return pred
    
    def __len__(self):
        return len(self.datalist)
    
    def __next__(self):
        for i in range(self.__len__):
            return self.__getitem__(i)
    
class poscarLoader():
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not such label file in {path}")
        self.path = path
    
    def load(self, name):
        """
        read the POSCAR or CONTCAR of VASP FILE
        and return the data position
        """
        abs_path = f"{self.path}/{name}"
        with open(abs_path) as fr:
            comment = fr.readline().split("\x00")[0]
            line = fr.readline()
            scale_length = float(_clean(line)[0])
            lattice = []
            for i in range(3):
                lattice.append(_clean(fr.readline()).astype(float))
            lattice = np.array(lattice)
            ele_name = _clean(fr.readline())
            counts = _clean(fr.readline()).astype(int)
            ele_num = dict(zip(ele_name, counts))
            fr.readline()
            fr.readline()
            positions = {}
            for ele in ele_name:
                position = []
                for _ in range(ele_num[ele]):
                    line = _clean(fr.readline())
                    position.append(line[:3].astype(float))
                positions[ele] = np.asarray(position)
        info = {'scale': scale_length, 
                'lattice': lattice, 
                'ele_num': ele_num,
                'ele_name': tuple(ele_name),
                'comment': comment}
        return info, positions
    
    def save(self, info):
        pass

class Logger():
    def __init__(self, path, elem = ("O", "H"), split = (0,3)):
        self.logger = self.get_logger(path)
        self.elem = elem
        self.split = [f"{split[i]}-{split[i+1]}" for i in range(len(split)-1)]

    def info(self, *arg, **args):
        self.logger.info(*arg, **args)

    def epoch_info(self, epoch, train_dic, valid_dic):
        info = f"\nEpoch = {epoch}" + "\n"
        info += f"Max memory use={torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB" + "\n"

        info += f"Train info: loss = {train_dic['loss'].item():.15f}"
        dic = train_dic['count']
        for ele in self.elem:
            for split in self.split:
                key = f"{ele}-{split}-"
                info += "\n" + f"{ele}({split}A): ACC = {dic[f'{key}ACC'].item():10.8f} SUC = {dic[f'{key}SUC'].item():10.8f} TP = {dic[f'{key}TP'].item():8.0f} FP = {dic[f'{key}FP'].item():8.0f} FN = {dic[f'{key}FN'].item():8.0f}"

        info += "\n" + f"Valid info: loss = {valid_dic['loss'].item():.15f}"
        dic = valid_dic['count']
        for ele in self.elem:
            for split in self.split:
                key = f"{ele}-{split}-"
                # O(0-3A):	accuracy=nan	success=1.0000	TP=0, FP=0, FN=0
                info += "\n" + f"{ele}({split}A): ACC = {dic[f'{key}ACC'].item():10.8f} SUC = {dic[f'{key}SUC'].item():10.8f} TP = {dic[f'{key}TP'].item():8.0f} FP = {dic[f'{key}FP'].item():8.0f} FN = {dic[f'{key}FN'].item():8.0f}"

        self.info(info)

    def test_info(self, test_dic):
        info = f"\nTesting info" + "\n"
        info += f"Max memory use={torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB" + "\n"

        info += f"Testing info: loss = {test_dic['loss'].item():.15f}"
        dic = test_dic['count']
        for ele in self.elem:
            for split in self.split:
                key = f"{ele}-{split}-"
                info += "\n" + f"{ele}({split}A): ACC = {dic[f'{key}ACC'].item():10.8f} SUC = {dic[f'{key}SUC'].item():10.8f} TP = {dic[f'{key}TP'].item():8.0f} FP = {dic[f'{key}FP'].item():8.0f} FN = {dic[f'{key}FN'].item():8.0f}"

        self.info(info)
    
    def get_logger(self, save_dir):
        filename = f"{time.strftime('%Y-%m-%d', time.localtime())}.log"
        logger_name = "main"
        log_path = os.path.join(save_dir, filename)
        # 记录器
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        # 处理器
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_path, when='D', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # 格式化器
        formatter = logging.Formatter(fmt='[{asctime} - {name} - {levelname:>8s}]: {message}', datefmt='%m/%d/%Y %H:%M:%S',
                                    style='{')
        # 给处理器设置格式
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # 给记录器添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

def _clean(line, splitter=' '):
    """
    clean the one line by splitter
    all the data need to do format convert
    ""splitter:: splitter in the line
    """
    data0 = []
    line = line.strip().replace('\t', ' ').replace('\x00','')
    list2 = line.split(splitter)
    for i in list2:
        if i != '':
            data0.append(i)
    temp = np.array(data0)
    return temp
