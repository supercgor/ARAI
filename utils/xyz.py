import numpy as np

def read(path: str) -> tuple[list[np.ndarray[np.float_]], list[list[str]]]:
    with open(path, "r") as f:
        ion_nums = int(f.readline().strip())
        _ = f.readline()
        molecules = []
        types = []
        charges = []
        ids = []
        index = None
        for _ in range(ion_nums):
            line = f.readline().strip().split()
            if line[-1] == index:
                types[-1].append(line[0])
                molecules[-1].append(line[1:4])
                charges[-1].append(float(line[4]))
                ids[-1].append(float(line[5]))
            else:
                types.append([line[0],])
                molecules.append([line[1:4],])
                charges.append([float(line[4])])
                ids.append([float(line[5])])

                index = line[-1]
        
        types = list(map(lambda x: np.array(x, dtype=np.str_), types))
        molecules = list(map(lambda x: np.array(x, dtype=np.float_), molecules))
        charges = list(map(lambda x: np.array(x, dtype=np.float_), charges))
        ids = list(map(lambda x: np.array(x, dtype=np.int_), ids))
            
    return types, molecules, charges, ids

def write(path: str, types: list[np.ndarray[np.str_]], molecules: list[list[np.ndarray[np.float_]]], charges: list[list[np.float_]] = None, ids: list[list[int]] = None):
    out = f"{sum(map(len, molecules))}\n\n"
    if ids is None:
        idx = 0
        if charges is None:
            for i, (type, molecule) in enumerate(zip(types, molecules)):
                for j, (t, m) in enumerate(zip(type, molecule)):
                    out += f"{t} {m[0]:11.8f} {m[1]:11.8f} {m[2]:11.8f} {idx:8d}\n"
                    idx += 1
        else:
            for i, (type, molecule, charge) in enumerate(zip(types, molecules, charges)):
                for j, (t, m, c) in enumerate(zip(type, molecule, charge)):
                    out += f"{t} {m[0]:11.8f} {m[1]:11.8f} {m[2]:11.8f} {c:11.8f} {idx:8d}\n"
                    idx += 1
    else:
        if charges is None:
            for i, (type, molecule, id) in enumerate(zip(types, molecules, ids)):
                for j, (t, m, idx) in enumerate(zip(type, molecule, id)):
                    out += f"{t} {m[0]:11.8f} {m[1]:11.8f} {m[2]:11.8f} {idx:8d}\n"
        else:
            for i, (type, molecule, charge, id) in enumerate(zip(types, molecules, charges, ids)):
                for j, (t, m, c, idx) in enumerate(zip(type, molecule, charge, id)):
                    out += f"{t} {m[0]:11.8f} {m[1]:11.8f} {m[2]:11.8f} {c:11.8f} {idx:8d}\n"
                
    with open(path, "w") as f:
        f.write(out)

