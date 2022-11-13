import os
import random
import numpy as np
from pathlib import Path
import shutil


def write_filelist(T=(180,)):
    root_path = Path(r'E:\data\afm_ml_3d')
    for t in T:
        filenames = list((root_path / 'data').glob(f'au_t{t}*'))
        print(f'T={t} {len(filenames)}')
        with open(root_path / f'FileList_T{t}', 'w') as file:
            for filename in filenames:
                file.write(f'{filename.stem}\n')


def split_filelist(T=(180,)):
    root_path = r'E:\data\afm_ml_3d'
    filenames = []
    for t in T:
        with open(os.path.join(root_path, f'FileList_T{t}')) as file:
            filenames.extend([line.strip() for line in file.readlines()])
    random.shuffle(filenames)
    split = [14000, 15966, 15966]
    index = 0
    for i, name in enumerate(['train_FileList', 'val_FileList', 'test_FileList']):
        with open(os.path.join(root_path, name), 'w') as file:
            while True:
                if index == split[i]:
                    break
                file.write(filenames[index] + '\n')
                index += 1


def split_exist_file():
    root_path = Path(r"E:\data\afm_ml_ice\data")
    filenames = []
    for filename in root_path.glob("*"):
        filenames.append(filename.name)

    split = [3, 1, 1]
    base = 1

    save_dir = Path(r"E:\data\afm_ml_ice")
    with open(save_dir / "train_FileList", 'w') as file:
        for filename in filenames[base*0:base*3]:
            file.write(f"{filename}\n")
    with open(save_dir / "val_FileList", 'w') as file:
        for filename in filenames[base*3:base*4]:
            file.write(f"{filename}\n")
    with open(save_dir / "test_FileList", 'w') as file:
        for filename in filenames[base*4:base*5]:
            file.write(f"{filename}\n")


def use_small_filelist():
    root_dir = Path(r"E:\data\afm_ml_ice\afm_3a")

    frac = 1 / 3
    keys = ['train', 'val', 'test']
    for key in keys:
        with open(root_dir / f"{key}_FileList", 'r') as file:
            filelist = []
            for line in file:
                filelist.append(line.strip())
        small_filelist = filelist[:int(len(filelist) * frac)]
        with open(root_dir / f"{key}_FileList_small", 'w') as file:
            for filename in small_filelist:
                file.write(f"{filename}\n")


def get_filelist():
    root_dir = Path(r"E:\data\afm_ml_ice\test")
    with open(root_dir / f"test_FileList", 'w') as file:
        for filename in root_dir.glob("*"):
            if filename.is_dir():
                file.write(f"{filename.name}\n")


def slice2data():
    root_dir = Path(r"E:\data\afm_ml_ice")
    save_dir = Path(r"E:\data\afm_ml_ice\test_data")
    with open(root_dir / "test_FileList", 'w') as file:
        for file_dir in (root_dir / "test").glob("*"):
            file_pre = file_dir.name
            for pic_dir in (file_dir / "bulk_slice_box").glob("*"):
                if pic_dir.is_dir():
                    file_suf = pic_dir.name
                    filename = f"{file_pre}___{file_suf}"
                    shutil.copytree(pic_dir, save_dir / filename)
                    file.write(f"{filename}\n")


def data2slice():
    root_dir = Path(r"E:\data\afm_ml_ice\test_result")
    save_dir = Path(r"E:\data\afm_ml_ice\test_poscar")
    for file in root_dir.glob("*"):
        filename = file.name
        file_pre, file_suf = filename.split("___")
        (save_dir / file_pre).mkdir(exist_ok=True)
        shutil.copy(file, save_dir / file_pre / file_suf)


if __name__ == '__main__':
    # split_exist_file()
    # use_small_filelist()
    # get_filelist()
    # slice2data()
    data2slice()
