import os
os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import firedrake
import icepack
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    weertman_sliding_law as m
)
import icepack.models.friction
import icepack.plot
from firedrake import sqrt, inner
from firedrake import div
from firedrake.plot import triplot
import tqdm
import numpy as np
from firedrake import conditional


def acc_rates(params):
    Lx, Ly = 50e3, 12e3
    nx, ny = 48, 32
    mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly)

    Q = firedrake.FunctionSpace(mesh, "CG", 2)
    V = firedrake.VectorFunctionSpace(mesh, "CG", 2)

    x, y = firedrake.SpatialCoordinate(mesh)
    xfunc = firedrake.interpolate(x, Q)
    yfunc = firedrake.interpolate(y, Q)

    b_in, b_out = 200, -400
    sol_index = 0

    b = firedrake.interpolate(b_in - (b_in - b_out) * x / Lx, Q)

    s_in, s_out = 850, 50
    s0 = firedrake.interpolate(s_in - (s_in - s_out) * x / Lx, Q)

    h0 = firedrake.interpolate(s0 - b, Q)

    h_in = s_in - b_in
    δs_δx = (s_out - s_in) / Lx
    τ_D = -ρ_I * g * h_in * δs_δx
    # print(f"{1000 * τ_D} kPa")

    u_in, u_out = 20, 2400
    velocity_x = u_in + (u_out - u_in) * (x / Lx) ** 2

    u0 = firedrake.interpolate(firedrake.as_vector((velocity_x, 0)), V)

    ice_stream_sol = np.zeros((65,97,2))
    T = firedrake.Constant(255.0) 
    A = icepack.rate_factor(T)

    expr = 0.01*firedrake.cos(params[0]*x/Lx+params[1]) + 0.03
    C = firedrake.interpolate(expr, Q)
    test = np.vstack((xfunc.dat.data, yfunc.dat.data,  C.dat.data)).T
    test = test[(-test[:,1]).argsort()]
    test = test.reshape((65,97,3))
    for j in range(65):
        curr = np.vstack((test[j,:,0], test[j,:,1], test[j,:,2])).T
        curr = curr[curr[:,0].argsort()]
        curr = curr.T
        test[j,:,0] = curr[0,:]
        test[j,:,1] = curr[1,:]
        test[j,:,2] = curr[2,:]

    ice_stream_sol[:,:,0] =  test[:,:,2]

    p_W = ρ_W * g * firedrake.max_value(0, h0 - s0)
    p_I = ρ_I * g * h0
    ϕ = 1 - p_W / p_I

    def weertman_friction_with_ramp(**kwargs):
        u = kwargs["velocity"]
        h = kwargs["thickness"]
        s = kwargs["surface"]
        C = kwargs["friction"]

        p_W = ρ_W * g * firedrake.max_value(0, h - s)
        p_I = ρ_I * g * h
        ϕ = 1 - p_W / p_I
        return icepack.models.friction.bed_friction(
            velocity=u,
            friction=C * ϕ,
        )

    model_weertman = icepack.models.IceStream(friction=weertman_friction_with_ramp)
    opts = {"dirichlet_ids": [1], "side_wall_ids": [3, 4]}
    solver_weertman = icepack.solvers.FlowSolver(model_weertman, **opts)

    u0 = solver_weertman.diagnostic_solve(
        velocity=u0, thickness=h0, surface=s0, fluidity=A, friction=C
    )

    # Take initial glacier state and project it forward until it reaches a steady state
    num_years = 250
    timesteps_per_year = 2 # 6 months

    δt = 1.0 / timesteps_per_year
    num_timesteps = num_years * timesteps_per_year
  
    a_in = firedrake.Constant(1.7)
    δa = firedrake.Constant(-2.7)
    expr = a_in + δa * x / Lx - conditional(x / Lx > 0.5, 1, 0.0)
    a = firedrake.interpolate(expr, Q)
    # ice_stream_sol[:,:,index] = a.dat.data.reshape(65,97)
    
    h = h0.copy(deepcopy=True)
    u = u0.copy(deepcopy=True)

    for step in tqdm.trange(num_timesteps):
        h = solver_weertman.prognostic_solve(
            δt,
            thickness=h,
            velocity=u,
            accumulation=a,
            thickness_inflow=h0,
        )
        s = icepack.compute_surface(thickness=h, bed=b)

        u = solver_weertman.diagnostic_solve(
            velocity=u,
            thickness=h,
            surface=s,
            fluidity=A,
            friction=C,
        )
        if step==num_timesteps-1:
            # Velocity
            test = np.vstack((xfunc.dat.data, yfunc.dat.data, u.dat.data[:,0])).T
            test = test[(-test[:,1]).argsort()]
            test = test.reshape((65,97,3))
            for j in range(65):
                curr = np.vstack((test[j,:,0], test[j,:,1], test[j,:,2])).T
                curr = curr[curr[:,0].argsort()]
                curr = curr.T
                test[j,:,0] = curr[0,:]
                test[j,:,1] = curr[1,:]
                test[j,:,2] = curr[2,:]
            ice_stream_sol[:,:,1]  = test[:,:,2]

    return ice_stream_sol   

samples = 8 #20
slope = np.linspace(1, 50, samples) #50, 500
intercept = np.linspace(0.1, 6, samples)

ice_stream_sol = np.zeros((samples**2, 65,97, 2))

sol_index = 0
for i in tqdm.trange(samples):
        for j in range(samples):
            params = [slope[i], intercept[j]]
            ice_stream_sol[sol_index,:,:,:] = acc_rates(params)
            sol_index+=1

np.save(f"/home/ak2152@ad.eng.cam.ac.uk/mnt/Code/fourier_neural_operator-master/inverse/friction_cos2.npy", ice_stream_sol)

# ice_stream_sol = acc_rates([0.1, 0.4])

# White Gaussian noise
# mean = 0
# stddev = 3

# # generate white Gaussian noise with the same shape as the data array
# noisy_data = np.zeros((65,97,10))
# for j in range(10):
#     noise = np.random.normal(mean, stddev, ice_stream_sol.shape[:2])
#     noisy_data[:,:,j] = ice_stream_sol[:,:,1] + noise

# # noisy_data = np.zeros((65,97,1))
# # noisy_data[:,:,0] = ice_stream_sol[:,:,1]
# # np.save(f"observed_melt.npy", noisy_data)
# np.save(f"observed_friction_cos_noisy.npy", noisy_data)


