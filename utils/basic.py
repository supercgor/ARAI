import numpy as np


def read_file(file_name):
    data = []
    with open(file_name) as fr:
        for line in fr:
            fn = line.strip().replace('\t', ' ')
            fn2 = fn.split(" ")
            if fn2[0] != '':
                data.append(fn2[0])
    return data


def clean(line, splitter=' '):
    """
    clean the one line by splitter
    all the data need to do format convert
    ""splitter:: splitter in the line
    """
    data0 = []
    line = line.strip().replace('\t', ' ')
    list2 = line.split(splitter)
    for i in list2:
        if i != '':
            data0.append(i)
    temp = np.array(data0)
    return temp


def read_POSCAR(file_name):
    """
    read the POSCAR or CONTCAR of VASP FILE
    and return the data position
    """
    with open(file_name) as fr:
        comment = fr.readline()
        line = fr.readline()
        scale_length = float(clean(line)[0])
        lattice = []
        for i in range(3):
            lattice.append(clean(fr.readline()).astype(float))
        lattice = np.array(lattice)
        ele_name = clean(fr.readline())
        counts = clean(fr.readline()).astype(int)
        ele_num = dict(zip(ele_name, counts))
        fr.readline()
        fr.readline()
        positions = {}
        for ele in ele_name:
            position = []
            for _ in range(ele_num[ele]):
                line = clean(fr.readline())
                position.append(line[:3].astype(float))
            positions[ele] = np.asarray(position)
    info = {'comment': comment, 'scale': scale_length, 'lattice': lattice, 'ele_num': ele_num,
            'ele_name': tuple(ele_name)}
    return info, positions
