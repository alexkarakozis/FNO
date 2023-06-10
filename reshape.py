from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dolfin import Mesh
from tqdm import tqdm
import os


dir_path = "Samples/"

files = os.listdir(dir_path)

solutions = np.zeros((len(files), 78, 438, 20))
# Use this to terminal to limit CPU core usage
# export OMP_NUM_THREADS=2

# Load the mesh with hole from file
mesh = Mesh("rectangle_with_hole.xml")
# mesh = Mesh("rectangle_without_hole.xml")
# plot(mesh)
# plt.savefig('mesh.png')
# exit()

##########################################################################
# Fix square hole in data and prepare for the operator
##########################################################################
coord = np.array(mesh.coordinates())
x_coord = coord[:,0]
y_coord = coord[:,1]

# Load the mesh without hole from file
mesh_without = Mesh("rectangle_without_hole.xml")
coord_without = np.array(mesh_without.coordinates())

for f in range(len(files)):

    # load file
    vel_cyl_with = np.load(f"Samples/{files[f]}")
    # Load data with hole
    # vel_cyl_with = np.load('Samples/vel_square_data_0.001_0.5_asym_up_high.npy')
    # vel_cyl_with = np.load('vel_sq_without.npy')

    # Get square coordinates and generate zeros
    square_index = 324
    x_square = coord_without[-square_index:,0]
    y_square = coord_without[-square_index:,1]
    square_zeros = np.zeros(square_index)

    # Combine all data
    x_coord_total = np.concatenate((x_coord, x_square))
    y_coord_total = np.concatenate((y_coord, y_square))
    total_vel = np.zeros((vel_cyl_with.shape[0]+square_index, vel_cyl_with.shape[1]))
    for i in range(total_vel.shape[1]):
        total_vel[:,i] = np.concatenate((vel_cyl_with[:,i], square_zeros))

    # Store solutions
    square_sol = np.zeros((78,438,20))
    # Sort coordinates grid so that plt imshow works (from top left to bottom right)
    for j in range(20):
        coord_total = np.vstack((x_coord_total, y_coord_total, total_vel[:,j])).T
        # coord_total = np.vstack((coord_without[:,0], coord_without[:,1], total_vel[:,j])).T
        coord_total = coord_total[(-coord_total[:, 1]).argsort()]  # sort by y
        coord_total = coord_total.reshape((78, 438, 3))

        for i in range(78):
            curr = np.vstack((coord_total[i,:,0], coord_total[i,:,1], coord_total[i,:,2])).T
            curr = curr[curr[:,0].argsort()] # sort by x
            curr = curr.T
            coord_total[i,:,0] = curr[0,:]
            coord_total[i,:,1] = curr[1,:]
            coord_total[i,:,2] = curr[2,:]

        square_sol[:,:,j] = coord_total[:,:,2]
        # sc = plt.scatter(coord_total[0,:], coord_total[1,:], c=coord_total[2,:])
        # plt.clf()
        # sc = plt.imshow(coord_total[:, :, 2])
        # plt.colorbar(sc)
        # plt.savefig(f'solution_{j}.png')
    solutions[f,:,:,:] = square_sol

# np.save("Code/fourier_neural_operator-master/Solutions/sol_square_data_0.001_0.5_asym_up_high.npy", square_sol)
np.save("Code/fourier_neural_operator-master/Solutions/solutions_total.npy", solutions)
##########################################################################
# END: Fix square hole in data and prepare for the operator
##########################################################################
