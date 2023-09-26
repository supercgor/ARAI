# %%
import torch
from torch import nn
import torch.nn.functional as F

net = "unet_tune_v0"
filelist = "../data/bulkice/test.filelist"
# source_dir = f"../data/bulkice/result/{net}"
source_dir = f"../data/bulkice/result/{net}_repair"
target_dir = "../data/bulkice/label"

# %%
from utils import poscar
from utils.metrics import analyse_cls

with open(filelist, "r") as f:
    files = f.readlines()

files = [file.strip() for file in files]
analizer = analyse_cls()

for file in files:
    predict = poscar._load_poscar(f"{source_dir}/{file}.poscar")
    target = poscar._load_poscar(f"{target_dir}/{file}.poscar")
    match = analizer.match(predict['pos'], target['pos'])
    analizer.elm += match
# %%
match = analizer.summary()
met = match.get_met()
# %%
print(met)
# %%
