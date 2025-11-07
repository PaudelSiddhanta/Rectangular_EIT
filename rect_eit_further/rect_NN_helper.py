
import torch
import numpy as np
from rect_grid import Node, GridStructure
import solve_linear as sl
import torch.nn as nn
import copy


class CustomCosineAnnealingWarmRestarts:
    def __init__(self, T_0, initial_lr, T_mult=1, eta_min=1e-8, eta_max_factor=0.08):
        """
            implements a cosisne annealing function
            T_0 = cycle length
            initial_lr = initial learning rate that you start with
            T_mult = if we want to increase the cycle length
            eta_min = the minimum value of learning rate we will allow our scheduler to have
            eta_max_factor = after every cycle, we wish to decrease the max learning rate by a factor of eta_max_factor
                            actually we set the base learning rate(the rate we start with to be initial_lr) and then every cycle later
                            the base_lr= eta_max_factor * base_lr

        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max_factor = eta_max_factor
        self.t = 0
        self.cycle = 0
        self.T_cur = T_0
        self.base_lr = initial_lr

    def step(self):

        """
            step function is called to get the value of the next learning rate according to the cossine annealing formula

        """
        if self.t >= self.T_cur:
            self.cycle += 1
            self.base_lr *= self.eta_max_factor
            self.T_cur = self.T_0 * (self.T_mult ** self.cycle)
            self.t = 0
        lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * self.t / self.T_cur))
        self.t += 1
        return lr

    def reset(self, new_base_lr):
        """
            reset function resets the base learning rate that the function is currently working with to change it to a new one
        """
        self.base_lr = new_base_lr
        self.cycle = 0
        self.T_cur = self.T_0
        self.t = 0


#  Helper Functions
def generate_dtn_data(grid, batch_size,device):
    """
        for a rectangular grid "grid", of size n by n, we generally generate 4n pairs of DtN data.
        But we can also choose to generate less than 4n pairs of data by using batch_size
        batch_size <= grid.n
    """
    dirichlet = torch.zeros((batch_size, 4*grid.n), dtype=torch.float64).to(device)
    neumann = torch.zeros((batch_size, 4*grid.n), dtype=torch.float64).to(device)

    for k in range(min(batch_size, 4*grid.n)):            
        dirichlet_data = {idx: 1.0 if idx == grid.boundary_index[k] else 0.0
                          for idx in grid.boundary_index}
        potentials, neumann_data = grid.solve_forward_problem(dirichlet_data)
        dirichlet[k, k] = 1.0
        for j, idx in enumerate(grid.boundary_index):
            neumann[k, j] = neumann_data[idx]

    data = torch.cat((dirichlet, neumann), dim=1)
    return data

def loss_function(device,output, grid, alpha=1.0):
    "compute the boundary, interior and total loss of the forward pass"
    batch_size = output.size(0)
    interior = output[:, :-4*grid.n].to(device)
    boundary = output[:, -4*grid.n:].to(device)
    interior_loss = alpha * torch.sum(interior ** 2) / (2 * batch_size)
    boundary_loss = torch.sum(boundary ** 2) / (2 * batch_size)
    total_loss = interior_loss + boundary_loss
    return total_loss, interior_loss, boundary_loss

def adam_with_grad_clip(device,param, grad, alpha, m_t, v_t, beta1, beta2, eps, time_step, max_grad_norm=1.0):
    "implements gradient desccent using adam optimization method"
    with torch.no_grad():
        if grad is None:
            grad = torch.zeros_like(param).to(device)
        grad_norm = torch.norm(grad, p=2)
        if grad_norm > max_grad_norm:
            grad = grad * max_grad_norm / grad_norm
        m_t1 = beta1 * m_t + (1 - beta1) * grad
        v_t1 = beta2 * v_t + (1 - beta2) * (grad ** 2)
        m_corr = m_t1 / (1 - beta1 ** time_step)
        v_corr = v_t1 / (1 - beta2 ** time_step)
        delta = alpha * m_corr / (torch.sqrt(v_corr) + eps)
        param = param - delta
        return param, m_t1, v_t1
