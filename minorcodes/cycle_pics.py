# %%
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import Compose
from model import build_cyc_model
from datasets import read_pic, PixelShift, Blur, ColorJitter, CutOut, Noisy
import os

model_path = "model/pretrain/cyclenet_v1/G_T2S_CP15_LOSS0.2132.pkl"
dir_path = "../data/testice/S2T"
dir_path = "../data/testice/T2S"
index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# %%

cyclenet= build_cyc_model()
cyclenet.load_state_dict(torch.load(model_path))
cyclenet.cuda()

pics_list = os.listdir(dir_path)
if "result" in pics_list:
    pics_list.remove("result")

trans = Compose([])

#trans = Compose([Blur(sigma = 0.5), ColorJitter(C = [0.7, 1.0]), Noisy(mode=["add"], intensity = 0.02)])

# trans = Compose([PixelShift(fill = None), 
#                          Blur(), 
#                          ColorJitter(B = [0.8, 1.0], C = [0.7, 1.0]), 
#                          Noisy(mode=["add"], 
#                                intensity = 0.05)])

# %%
cyclenet.eval().requires_grad_(False)
for pic_name in pics_list:
    pics = read_pic(f"{dir_path}/{pic_name}", index)
    pics = trans(pics)
    pics = pics.permute(1, 0, 2, 3)
    pics = pics[None, ...].cuda()
    result = cyclenet(pics).sigmoid()
    print(result.shape)

    os.makedirs(f"{dir_path}/result/{pic_name}", exist_ok=True)
    result = result[0].detach().cpu()
    pics = result.permute(1, 0, 2, 3)
    for i, pic in enumerate(pics):
        pic = pic.transpose(1, 2)
        pic = torch.flip(pic, [1])
        save_image(pic, f"{dir_path}/result/{pic_name}/{i}.png")
# %%
