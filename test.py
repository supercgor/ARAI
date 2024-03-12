from utils.const import water_molecule
from utils.lib import encodewater
from ase import Atoms
from ase.visualize import view
from matplotlib import pyplot as plt


fig = plt.figure(figsize=(15,12))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
for i in range(3):
    ax.quiver(0, 0, 0, *water_molecule[i], color='r')
    
code = encodewater(water_molecule.reshape(-1, 9)).reshape(3, 3)
for i in range(3):
    ax.quiver(0, 0, 0, *code[i], color='b')

plt.show()
