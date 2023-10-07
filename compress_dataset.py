import argparse
import cv2
import h5py
import numpy as np
import os
import tqdm
from utils import poscar

def parse_command_line_args():
    parser = argparse.ArgumentParser(description='This code is to compact the raw dataset into one file, the format is hdf5, which allows us to parallel read the folder faster.')
    parser.add_argument('input', type=str, help='Path of the raw dataset folder', default='testingdataset')
    parser.add_argument('-o','--output', type=str, help='Path of the output file', default='')
    parser.add_argument('-l', '--label', type=str, help='Whether to include the label (y/n)', default="y")
    parser.add_argument('-s', '--split', type=str, help='Whether to split the dataset into train/val/test (y/n)', default="n")
    return parser.parse_args()

args = parse_command_line_args()
args.label = args.label.lower().strip() in ['y', 'yes', 'ok', "true", "1", "t"]
args.split = args.split.lower().strip() in ['y', 'yes', 'ok', "true", "1", "t"]
input_dir: str = args.input.rstrip('/')
output_path: str = args.output if args.output else input_dir
if not output_path.endswith('.hdf5'):
   output_path += '.hdf5'

afm_dirnames = os.listdir(f"{input_dir}/afm") # *
afm_dirnames = [i for i in afm_dirnames if not i.startswith('.')]
afm_dirnames.sort()
if args.label:
    label_filenames = os.listdir(f"{input_dir}/label") # *.poscar
    label_filenames = [i for i in label_filenames if not i.startswith('.')]
    missed_label = [f"{i}.poscar" for i in afm_dirnames if f"{i}.poscar" not in label_filenames]
    missed_afm = [i.replace(".poscar", "") for i in label_filenames if i.replace(".poscar", "") not in afm_dirnames]
    if missed_label:
        print("Warning: the following .poscar are missed in the label folder:")
        print(*missed_label, sep=", ")
    if missed_afm:
        print("Warning: the following dir are missed in the afm folder:")
        print(*missed_afm, sep=", ")
    if missed_label or missed_afm:
        print("Confirm to continue? (y/n)")
        inp = input().lower().strip()
        if inp not in ['y', 'yes', 'ok', "true", "1", "t"]:
            exit(0)
    for i in missed_label:
        afm_dirnames.remove(i)
        
    for i in missed_afm:
        label_filenames.remove(f"{i}.poscar")
        
    
def load_process(name: str) -> h5py.Group:
    imgs_index = os.listdir(f"{input_dir}/afm/{name}")
    imgs_index = [i[:-4] for i in imgs_index if i.endswith('.png')]
    imgs_index.sort(key=lambda x: int(x))
    imgs = []
    for img_index in imgs_index:
        img = cv2.imread(f"{input_dir}/afm/{name}/{img_index}.png", cv2.IMREAD_GRAYSCALE)
        imgs.append(img)
    imgs = np.array(imgs)
    imgs = imgs[None, ...]
    imgs = imgs / 256
    group = h5file.create_group(name)
    gp_afm = group.create_dataset('afm', data=imgs)
    gp_afm.attrs['format'] = "CDHW"
    gp_afm.attrs['shape'] = str(imgs.shape)
    gp_afm.attrs['dtype'] = str(imgs.dtype)
    
    """
    "comment": comment,
    "scaling_factor": scaling_factor,
    "lattice": lattice,
    "ion": ion,
    "ion_num": ion_num,
    "selective_dynamics": selective_dynamics,
    "pos_mode": pos_mode,
    "pos": pos,
    "pos_dyn": pos_dyn,
    """
    if args.label:
        label = poscar.load(f"{input_dir}/label/{name}.poscar")
        gp_label = group.create_dataset('label', data=label['pos'])
        gp_label.attrs['comment'] = label['comment']
        gp_label.attrs['scaling_factor'] = label['scaling_factor']
        gp_label.attrs['lattice'] = label['lattice']
        gp_label.attrs['ion'] = label['ion']
        gp_label.attrs['ion_num'] = label['ion_num']
        gp_label.attrs['pos_mode'] = label['pos_mode']

if args.split:
    print("Enter the ratio of train/test (e.g. '0.8,0.2')")
    inp = input().strip().split(',')
    inp = [float(i) for i in inp]
    inp = [i / sum(inp) for i in inp]
    test_num = int(len(afm_dirnames) * inp[1])
    train_num = len(afm_dirnames) - test_num
    np.random.shuffle(afm_dirnames)
    train_names = afm_dirnames[:train_num]
    train_names.sort()
    test_names = afm_dirnames[train_num:]
    test_names.sort()
    with h5py.File(f"{output_path[:-5]}_train.hdf5", 'w') as h5file:
        print("Start processing training dataset...")
        for i in tqdm.tqdm(train_names):
            load_process(i)
        print(f"Successfully saved to '{output_path[:-5]}_train.hdf5'.")
        print("Done!")
    with h5py.File(f"{output_path[:-5]}_test.hdf5", 'w') as h5file:
        print("Start processing testing dataset...")
        for i in tqdm.tqdm(test_names):
            load_process(i)
        print(f"Successfully saved to '{output_path[:-5]}_test.hdf5'.")
        print("Done!")
else:
    with h5py.File(output_path, 'w') as h5file:
        print("Start processing...")
        for i in tqdm.tqdm(afm_dirnames):
            load_process(i)
        print(f"Successfully saved to '{output_path}'.")
        print("Done!")
    
        