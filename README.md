This repo contains some of the experiments associated with the paper :

## The discrete inverse conductivity problem solved by the weights of an interpretable neural network 
 - by Elena Beretta, Maolin Deng, Alberto Gandol, Bangti Jin.

## The link to the paper can be found [here](https://doi.org/10.1016/j.jcp.2025.114162).
## The detailed description of the approach is also described in Summer_work_report.pdf

In these experiments, we demonstrate the solution to Discrete Inverse Conductivity Problem of recovering the conductivity profile on network edges from the discrete
Dirichlet-to-Neumann map on a square lattice. The novelty of the approach lies in the fact that the
sought-after conductivity is not provided directly as the output of the NN but is instead encoded in
the weights of the post-trainig NN in the second layer. Hence the weights of the trained NN acquire
a clear physical meaning, which contrasts with most existing neural network approaches, where
the weights are typically not interpretable. This work represents a step toward designing NNs with
interpretable post-training weights.

## File Structure and General Description



This repository consists of code only for the rectangular lattices. 

- The structure of the rectangular lattices can be found in rect_grid.py.
- Functionalities like assigning conductivities, solving the forward problem (required for the generation of the data) and visualizing the networks with additional functions are present in rect_grid.py.
- Additional funcitons to solve the linear system using LU decomposition is implemented in solve_linear.py and 
- The initial attempt was done by using Autograd functionality from Pytorch. A clean(but unsuccessful) implementation with proper description can be found in Clean_rectangular_NN.ipynb.
- Due to the failure of the initial approach, a simple function optimization approach was performed using minimize function from scipy.minimize as a blackbox. The implementation can be seen in function_optimization_testing_with_small_dTNvalues_n2.ipynb and function_optimization_testing_with_small_dTNvalues_n3.ipynb
Partial success was achieved. Also, experiments with less than 4n DtN pairs were conducted using this optimization method. One can also observe the weight matrix more clearly in symbolic form in these notebooks.
- The issues with the Neural network was due to the use of Autograds because of the dependence of parameters. Autograd fails in cases where the parameters of the same layer are dependent on one another. Hence, the training loop was changed, ADAM optimizer was used along with cosine annealing for learning rate scheduling to get better performance. The remaining notebooks in this folder show different experiments and improvements in the loss and extracted conductivity values compared to the previous attempts.
- rect_non_lin.ipynb is an attempt to perform the same experiment but for the non linear case where there is an extra cubic term of potential while trying to apply Kirchoff's laws.
- Inside the rect_eit_firther, there are some more experiments, run on HPC, with higher epochs and for different n values.
- Some experiments were conducted using NYUAD HPC resources.

