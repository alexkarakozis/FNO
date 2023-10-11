# Fourier Neural Operator (FNO) for ice sheet modelling

### Build Requirements & References

This repository is based on
- the paper: https://arxiv.org/abs/2010.08895
- the github repository: https://github.com/neuraloperator/neuraloperator/tree/master
- ice sheet simulation software: https://icepack.github.io/ 
  
which develop and implement the FNO, respectively.

### Summary
In this repository, the Fourier Neural Operator (FNO) is examined in the context of ice sheet modelling as a potential replacement or complement to traditional solvers. We conduct simulations using a traditional solver to model the evolution of an ice stream, focusing on three parameters; melt rate, rate factor, and sliding friction exponent. We train the FNO using data from the traditional solver and solve the forward problem of predicting the steady-state ice stream velocity field. The FNO exhibits low error and makes fast predictions, requiring only a single forward pass. We also apply the FNO in an inverse Bayesian problem, using the preconditioned Crank-Nicolson Markov chain Monte Carlo (pCN-MCMC) algorithm for parameter estimation. The FNO serves as a surrogate model, replacing the slower traditional solver. We assess convergence and the mixing properties of the chain by visual inspection of the trace and auto-correlation plots and by calculating the effective sample size. The samples generated accurately estimate parameters without significant FNO hyperparameter tuning and with a limited number of training samples. The FNO demonstrates faster computation times than the solver with slightly worse levels of accuracy. Testing the FNO on data outside the training range indicates a deterioration in estimation accuracy with increasing distance from the training range. The inverse crime problem suggests that overfitting and poor generalization are potential risks in the simulations due to using the solver to generate both the training data and the noisy observations. We train the FNO with low-resolution data and use high-resolution sub-sampled noisy observations. In this way, we verify that the inverse crime problem associated error is low and therefore can be ignored in these simulations. Next, we carry out multiple parameters inference. The FNO successfully performs multiple parameters inference, estimating the melt rate and rate factor simultaneously. It exhibits faster computation times compared to the solver and low error. We investigate the multiple parameters inference using the FNO by incrementally increasing the noise in the observations. The FNO converges near the true parameter values and the error experienced increases with increasing noise levels. The error is not however that large even for the highest noise level indicating FNO’s potential for reducing parameter estimation time. Recommendations for further research include exploring complex parameter functions for estimation, introducing non-zero mean and correlation in the prior used for proposals in the pCN-MCMC algorithm, performing additional convergence tests and investigating further the effect of using the traditional solver for generation of both training and observation data. The limitations of uniform discretization due to the Fast Fourier Transform (FFT) used by the FNO can be addressed by exploring the Geo-FNO which can be applied to complex meshes. We conclude that the FNO can accurately and rapidly predict solutions and estimate parameters of the ice stream model. In contrast to the solver, the FNO does not need retraining if initial conditions change slightly, making it more adaptable. It can possibly be trained on sensor data without knowledge of the underlying PDE, allowing for scientific discovery. The FNO can complement traditional solvers in ice sheet modelling as it is computationally less expensive and highly accurate.  

### The pcN-MCMC algorithm

The pCN-MCMC algoritm used for parameters estimation in the inverse Bayesian problem is presented below. In the pseudo-code below, $y$ are observed data and $u_{new}$ are the predicted data denoted by $f$ in section 7.2.

<img width="684" alt="image" src="https://github.com/alexkarakozis/FNO/assets/69156399/bfd6505c-2e3c-4e44-831e-641c89a3c57f">

### Results

### Limitations

### Conclusions






