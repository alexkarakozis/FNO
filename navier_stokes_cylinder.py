"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""
from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dolfin import Mesh
from tqdm import tqdm

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
# coord = np.array(mesh.coordinates())
# x_coord = coord[:,0]
# y_coord = coord[:,1]

# # Load the mesh without hole from file
# mesh_without = Mesh("rectangle_without_hole.xml")
# coord_without = np.array(mesh_without.coordinates())

# # Load data with hole
# vel_cyl_with = np.load('vel_sq_with.npy')
# # vel_cyl_with = np.load('vel_sq_without.npy')

# # Get square coordinates and generate zeros
# square_index = 324
# x_square = coord_without[-square_index:,0]
# y_square = coord_without[-square_index:,1]
# square_zeros = np.zeros(square_index)

# # Combine all data
# x_coord_total = np.concatenate((x_coord, x_square))
# y_coord_total = np.concatenate((y_coord, y_square))
# total_vel = np.zeros((vel_cyl_with.shape[0]+square_index, vel_cyl_with.shape[1]))
# for i in range(total_vel.shape[1]):
#     total_vel[:,i] = np.concatenate((vel_cyl_with[:,i], square_zeros))

# # Store solutions
# square_sol = np.zeros((78,438,20))
# # Sort coordinates grid so that plt imshow works (from top left to bottom right)
# for j in range(20):
#     coord_total = np.vstack((x_coord_total, y_coord_total, total_vel[:,j])).T
#     # coord_total = np.vstack((coord_without[:,0], coord_without[:,1], total_vel[:,j])).T
#     coord_total = coord_total[(-coord_total[:, 1]).argsort()]  # sort by y
#     coord_total = coord_total.reshape((78, 438, 3))

#     for i in range(78):
#         curr = np.vstack((coord_total[i,:,0], coord_total[i,:,1], coord_total[i,:,2])).T
#         curr = curr[curr[:,0].argsort()] # sort by x
#         curr = curr.T
#         coord_total[i,:,0] = curr[0,:]
#         coord_total[i,:,1] = curr[1,:]
#         coord_total[i,:,2] = curr[2,:]

#     square_sol[:,:,j] = coord_total[:,:,2]
#     # sc = plt.scatter(coord_total[0,:], coord_total[1,:], c=coord_total[2,:])
#     # plt.clf()
#     # sc = plt.imshow(coord_total[:, :, 2])
#     # plt.colorbar(sc)
#     # plt.savefig(f'solution_{j}.png')

# np.save("sol_square.npy", square_sol)
# exit()
##########################################################################
# END: Fix square hole in data and prepare for the operator
##########################################################################

samples = 3                                 # samples to generate
T = 1.0             #5.0                     # final time
num_steps = 4000    #5000              # number of time steps
dt = T / num_steps                           # time step size
mus = np.linspace(0.001, 0.003, samples)     # dynamic viscosity
rhos = np.linspace(0.5, 1, 6) #1  # density

# Store solutions to numpy array
vel_square_data = np.zeros((np.array(mesh.coordinates()).shape[0],20))

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundaries
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
# walls    = 'near(x[1], 0) || near(x[1], 0.41)'
walls    = 'near(x[1], 0) || near(x[1], 0.4)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

# Define inflow profile
# inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')
inflow_profile = ('4.0*2*x[1]*(0.46 - x[1]) / pow(0.46, 2)', '0')
# inflow_profile = ('4.0*1.5*x[1]*(0.4 - x[1]) / pow(0.4, 2)', '0')

# Define boundary conditions
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]


for j in tqdm(range(len(mus))):
    for k in range(len(rhos)):
        curr_mu = mus[j]
        curr_rho = rhos[k]
        
        # Define trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)

        # Define functions for solutions at previous and current time steps
        u_n = Function(V)
        u_  = Function(V)
        p_n = Function(Q)
        p_  = Function(Q)

        # Define expressions used in variational forms
        U  = 0.5*(u_n + u)
        n  = FacetNormal(mesh)
        f  = Constant((0, 0))
        k  = Constant(dt)
        rho = Constant(curr_rho)
        mu = Constant(curr_mu)

        # Define symmetric gradient
        def epsilon(u):
            return sym(nabla_grad(u))

        # Define stress tensor
        def sigma(u, p):
            return 2*mu*epsilon(u) - p*Identity(len(u))

        # Define variational problem for step 1
        F1 = rho*dot((u - u_n) / k, v)*dx \
        + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
        + inner(sigma(U, p_n), epsilon(v))*dx \
        + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
        - dot(f, v)*dx
        a1 = lhs(F1)
        L1 = rhs(F1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

        # Define variational problem for step 3
        a3 = dot(u, v)*dx
        L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

        # Assemble matrices
        A1 = assemble(a1)
        A2 = assemble(a2)
        A3 = assemble(a3)

        # Apply boundary conditions to matrices
        [bc.apply(A1) for bc in bcu]
        [bc.apply(A2) for bc in bcp]

        # Time-stepping
        t = 0
        # save index
        i = 0

        for n in tqdm(range(num_steps)):

            # Update current time
            t += dt

            # Step 1: Tentative velocity step
            b1 = assemble(L1)
            [bc.apply(b1) for bc in bcu]
            solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

            # Step 2: Pressure correction step
            b2 = assemble(L2)
            [bc.apply(b2) for bc in bcp]
            solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

            # Step 3: Velocity correction step
            b3 = assemble(L3)
            solve(A3, u_.vector(), b3, 'cg', 'sor')

            # Plot solution
            if n%200 == 0 and n >= 0:
                u_values, u_gradients = u_n.split()
                u_values = u_values.compute_vertex_values()
                vel_square_data[:,i] = np.array(u_values)
                i+=1
                # print(u_values.shape)
                # plt.clf()
                # sc = plot(u_, title='Velocity')
                # plt.colorbar(sc)
                # plot(p_, title='Pressure')
                # plt.savefig(f'vel_square_test{n}.png')

            # Update previous solution
            u_n.assign(u_)
            p_n.assign(p_)

            # Update progress bar
            # print('u max:', u_.vector().get_local().max())
            
        np.save(f"Samples/vel_square_data_{curr_mu}_{curr_rho}_asym_up_high.npy", vel_square_data[:,:])

# np.save("vel_square_data.npy", vel_square_data)