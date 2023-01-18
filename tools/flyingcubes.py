import numpy as np
from skimage import measure
from stl import mesh
import argparse
import logging
import os.path as path


# x_range = np.linspace(start=-np.pi, stop=np.pi, num=samples)
x_range = np.arange(start=-50, stop=+50, step=1)
y_range = np.arange(start=-50, stop=+50, step=1)
z_range = np.arange(start=0, stop=+200, step=1)

xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
voxel_grid = np.ones((len(x_range), len(y_range), (len(z_range))), dtype=float) * -1

for i in range(10):
    size = np.random.choice([3, 5, 7])
    size2 = size // 2
    x_center = np.random.randint(low=0+size2, high=10-size2)
    y_center = np.random.randint(low=0+size2, high=10-size2)
    z_center = np.random.randint(low=0+size2, high=50-size2)
    for x in range(x_center-size2, x_center+size2):
        for y in range(y_center-size2, y_center+size2):
            for z in range(z_center-size2, z_center+size2):
                voxel_grid[x, y, z] = 1.

# padding = 5
# voxel_grid = np.abs(voxel_grid)
# voxel_grid = np.pad(voxel_grid, padding, 'constant', constant_values=0.)

np.save('flyingcubes.npy', voxel_grid)
