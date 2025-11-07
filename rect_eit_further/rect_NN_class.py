import torch
import numpy as np
from rect_grid import Node, GridStructure
import solve_linear as sl
import torch.nn as nn
import copy

# SquareEITNN Class
class SquareEITNN(nn.Module):
    def __init__(self, grid,device):
        super(SquareEITNN, self).__init__()
        self.grid = grid
        self.n = grid.n
        self.input_size = 8 * self.n
        self.hidden_size = self.n ** 2 + 8 * self.n
        self.output_size = self.n ** 2 + 4 * self.n
        self.interior_size = self.n ** 2
        self.neumann_copy_start = self.n ** 2 + 4 * self.n
        self.device = device

        self.W1 = nn.Parameter(torch.zeros(self.hidden_size, self.input_size, dtype=torch.float64, requires_grad=False)).to(self.device)
        self.W1_fixed = torch.zeros(self.hidden_size, self.input_size, dtype=torch.float64).to(self.device)
        self.W1_fixed_mask = torch.ones(self.hidden_size, self.input_size, dtype=torch.bool).to(self.device)
        self.W1_fixed_mask[:self.interior_size, :4*self.n] = False

        self.W1__grad = torch.zeros(self.hidden_size, self.input_size, dtype=torch.float64, requires_grad=False).to(self.device)
        self.W2__grad = torch.zeros(self.output_size, self.hidden_size, dtype=torch.float64, requires_grad=False).to(self.device)
        self._initialize_W1_blocks()
        self._initialize_W2()

    def assign_gradients(self, x_data, h, y, alpha):
        with torch.no_grad():
            batch_size = y.shape[0]
            dL_dy = torch.zeros_like(y).to(self.device)
            dL_dy[:, :self.n**2] = alpha * y[:, :self.n**2] / batch_size
            dL_dy[:, self.n**2:] = y[:, self.n**2:] / batch_size

            dW2 = dL_dy.T @ h
            grad = {}
            w2_indices = [(i, j) for i in range(self.n**2) for j in self.grid.get_node_by_index(i).neighbors if i < j < self.interior_size + 4*self.n]
            for i, j in w2_indices:
                grad[(i, j)] = dW2[i, j] + dW2[j, i] - dW2[i, i] - dW2[j, j]
                self.W2__grad[i, j] = grad[(i, j)] / (1 + self.Beta[i, j]**2)

            dW1 = self.W2_dynamic.T @ dL_dy.T @ x_data
            for i in range(self.n**2):
                for j in range(4*self.n):
                    self.W1__grad[i, j] = dW1[i, j]

    def _initialize_W1_blocks(self):
        with torch.no_grad():
            self.W1[:self.interior_size, :4*self.n] = torch.randn(self.interior_size, 4*self.n, dtype=torch.float64).to(self.device)
           
            self.W1[:self.interior_size, 4*self.n:] = 0.0
            self.W1_fixed[:self.interior_size, 4*self.n:] = 0.0
           
            self.W1[self.interior_size:self.interior_size + 4*self.n, :4*self.n] = torch.eye(4*self.n, dtype=torch.float64).to(self.device)
            self.W1_fixed[self.interior_size:self.interior_size + 4*self.n, :4*self.n] = torch.eye(4*self.n, dtype=torch.float64).to(self.device)
           
            self.W1[self.interior_size:self.interior_size + 4*self.n, 4*self.n:] = 0.0
            self.W1_fixed[self.interior_size: self.interior_size + 4*self.n, 4*self.n:] = 0.0
           
            self.W1[self.interior_size + 4*self.n:, :4*self.n] = 0.0
            self.W1_fixed[self.interior_size + 4*self.n:, :4*self.n] = 0.0
           
            self.W1[self.interior_size + 4*self.n:, 4*self.n:] = torch.eye(4*self.n, dtype=torch.float64).to(self.device)
            self.W1_fixed[self.interior_size + 4*self.n:, 4*self.n:] = torch.eye(4*self.n, dtype=torch.float64).to(self.device)

    def _initialize_W2(self):
        size_out = self.output_size
        size_hidden = self.hidden_size
        self.Beta = nn.Parameter(torch.zeros(size_out, size_hidden, dtype=torch.float64, requires_grad=False)).to(self.device)
        self.W2_fixed = torch.zeros(size_out, size_hidden, dtype=torch.float64).to(self.device)
        self.W2_mask = torch.ones(size_out, size_hidden, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            for i in range(self.interior_size):
                node_i = self.grid.get_node_by_index(i)
                neighbor_indices = [j for j in node_i.neighbors if i < j < self.interior_size + 4*self.n]
                weights = torch.abs(torch.randn(len(neighbor_indices), dtype=torch.float64)).to(self.device)
                for k, j in enumerate(neighbor_indices):
                    self.Beta[i,j] = weights[k]
                    self.Beta[j, i] = weights[k]
                    self.W2_fixed[j, i] = weights[k]
                    self.W2_mask[i, j] = False
                self.Beta[i, i] = 0
                self.W2_fixed[i, i] = 0

            for i in range(self.interior_size + 4*self.n):
                self.Beta[i, i] = -torch.sum(torch.tensor([self.Beta[i, j] for j in range(self.interior_size + 4*self.n)], dtype=torch.float64).to(self.device))
                self.W2_fixed[i, i] = -torch.sum(torch.tensor([self.W2_fixed[i, j] for j in range(self.interior_size + 4*self.n)], dtype=torch.float64).to(self.device))

            for i in range(4*self.n):
                row = self.interior_size + i
                col = self.neumann_copy_start + i
                self.Beta[row, col] = 1.0
                self.W2_fixed[row, col] = 1.0

    def symmetrize_W2_after_training(self):
        with torch.no_grad():
            for i in range(self.interior_size):
                node_i = self.grid.get_node_by_index(i)
                neighbor_indices = [j for j in node_i.neighbors if i < j < self.interior_size + 4*self.n]
                for k, j in enumerate(neighbor_indices):
                    self.W2_fixed[i,j] = self.transform(self.Beta[i,j])
                    self.W2_fixed[j,i] = self.transform(self.Beta[i,j])
            for i in range(self.interior_size + 4*self.n):
                self.W2_fixed[i, i] = 0
                self.W2_fixed[i, i] = -torch.sum(torch.tensor([self.W2_fixed[i, j] for j in range(self.interior_size + 4*self.n)], dtype=torch.float64).to(self.device))

    def transform(self, x):
        return torch.arctan(x) + torch.pi/2

    def forward(self, x):
        self.W1_dynamic = torch.where(self.W1_fixed_mask, self.W1_fixed, self.W1).to(self.device)
        self.W2_dynamic = torch.where(self.W2_mask, self.W2_fixed, torch.arctan(self.Beta) + torch.pi/2).to(self.device)
        hid = torch.matmul(x, self.W1_dynamic.t())
        outp = torch.matmul(hid, self.W2_dynamic.t())
        return outp, hid


