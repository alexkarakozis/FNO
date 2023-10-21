# Fourier Neural Operator (FNO) for ice sheet modelling

### Summary
We examine the Fourier Neural Operator (FNO) in the context of ice sheet modelling as a potential replacement or complement to traditional solvers. We conduct simulations using a traditional solver to model the evolution of an ice stream, focusing on three parameters; melt rate, rate factor, and sliding friction exponent. We train the FNO using data from the traditional solver and solve the forward problem of predicting the steady-state ice stream velocity field. The FNO exhibits low error and makes fast predictions, requiring only a single forward pass. Next, we apply the FNO in an inverse Bayesian problem, using the preconditioned Crank-Nicolson Markov chain Monte Carlo (pCN-MCMC) algorithm for parameter estimation. The FNO serves as a surrogate model, replacing the slower traditional solver. We assess convergence and the mixing properties of the chain by visual inspection of the trace and auto-correlation plots. The samples generated accurately estimate parameters without significant FNO hyperparameter tuning and with a limited number of training samples. The FNO demonstrates faster computation times than the solver with slightly worse levels of accuracy. 

### The pcN-MCMC algorithm

In the pseudo-code, $y$ is the observed data, which is a noisy velocity field, and $u_{new}$ is the predicted data.

<img width="684" alt="image" src="https://github.com/alexkarakozis/FNO/assets/69156399/bfd6505c-2e3c-4e44-831e-641c89a3c57f">

### Example Results

The results for the melt rate parameter estimation are presented below.

<img width="941" alt="Screenshot 2023-10-13 at 14 55 52" src="https://github.com/alexkarakozis/FNO/assets/69156399/d8f046c8-a4e4-4a0a-8be1-0f8763873a9d">

<img width="941" alt="Screenshot 2023-10-13 at 14 55 40" src="https://github.com/alexkarakozis/FNO/assets/69156399/3762fd3e-b391-41c6-9ca1-c91694ca7664">


### Conclusions
The FNO can accurately and rapidly predict solutions and estimate parameters of the ice stream model. In contrast to the solver, the FNO does not need retraining if initial conditions change slightly, making it more adaptable. It can possibly be trained on sensor data without knowledge of the underlying PDE, allowing for scientific discovery. The FNO can complement traditional solvers in ice sheet modelling as it is computationally less expensive and highly accurate. The limitation of the FNO is that the Fast Fourier Transform (FFT) requires uniform meshes (recatngular grids).

Further information and details can be provided upon request.

### Build Requirements & References

This repository is an application of the FNO and is based on
- https://arxiv.org/abs/2010.08895
- https://github.com/neuraloperator/neuraloperator/tree/master

which develop and implement the FNO, respectively.

The ice sheet modelling is based on the ice sheet simulation software: 
- https://icepack.github.io/
