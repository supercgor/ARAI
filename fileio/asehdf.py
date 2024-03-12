from ase.atoms import Atoms, Atom
import numpy as np
from h5py import File

def write_array(file: File, name: str, array: np.ndarray, property: dict = None):
    dts = file.create_dataset(name, data=array)
    if property is not None:
        for key, val in property.items():
            dts.attrs[key] = val
    return dts

def write_dict(file: File, name: str, dic: dict, property: dict = None):
    group = file.create_group(name)
    for key, val in dic.items():
        group[key] = val
    if property is not None:
        for key, val in property.items():
            group.attrs[key] = val
    return group

def load_by_name(file: File, name: str):
    atom_keys = ['positions', 'numbers', 'cell', 'pbc']
    dic = dict(file[name].items())
    dic.update(file[name].attrs.items())
    atom_dic = {key: val[...] for key, val in dic.items() if key in atom_keys}
    info_dic = {key: val if isinstance(val, str) else val[...] for key, val in dic.items() if key not in atom_keys}
    info_dic['name'] = name
    atom_dic['info'] = info_dic
    atoms = Atoms(**atom_dic)
    return atoms

def list_names(file: File):
    return list(file.keys())