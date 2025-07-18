import math
import numpy as np
import torch
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.colors import Normalize
import solve_linear as sl


class Node:
    def __init__(self, i, j, n, index, is_boundary=False):
        """
        Initialize a node in the rectangular network.
        
        Args:
            i (int): x-coordinate (0 to n+1).
            j (int): y-coordinate (0 to n+1).
            n (int): Size of the grid (n x n interior nodes).
            index (int): Unique index for the node.
            is_boundary (bool): True if node is on the boundary.
        """
        self.i = i  # x-coordinate
        self.j = j  # y-coordinate
        self.n = n
        self.index = index
        self.is_boundary = is_boundary
        self.is_interior = not is_boundary
        # Compute Cartesian coordinates
        self.x = i
        self.y = j
        # Initialize neighbors list
        self.neighbors = []
        # Placeholder for potential
        self.potential = 0.0

    def add_neighbor(self, neighbor_index):
        """
        Add a neighbor's index to the neighbors list.
        
        Args:
            neighbor_index (int): Index of neighboring node.
        """
        self.neighbors.append(neighbor_index)

    def __repr__(self):
        return f"Node(i={self.i}, j={self.j}, boundary={self.is_boundary})"


class GridStructure:
    def __init__(self, n):
        """
        Initialize the rectangular network of size n x n interior nodes.
        Corner nodes (0,0), (0,n+1), (n+1,0), (n+1,n+1) are excluded.
        Interior nodes are indexed first (0 to n^2-1), followed by boundary nodes
        in cyclic order (n^2 to n^2+4n-1).
        
        Args:
            n (int): Number of interior nodes along each dimension.
        """
        self.n = n
        self.nodes = {}
        self.index_to_coords = {}  # Maps index to (i,j)
        self.edges = []  # List of tuples (x,y) where x<y are node indices
        self.conductivities = {}  # Maps each edge (x,y) to conductivity
        self.create_nodes()
        self.node_count = len(self.nodes)
        self.boundary_index = []
        self.interior_index = []
        self.get_boundary_index()
        self.get_internal_index()
        self.boundary_node_count = len(self.boundary_index)
        self.internal_node_count = len(self.nodes) - self.boundary_node_count
        self.assign_neighbors()
        self.create_edges()
        self.edge_count = len(self.edges)
        self.generate_conductivity()

    def create_nodes(self):
        """
        Create all nodes in the network, excluding corner nodes.
        Nodes are at (i,j) for i,j=0,...,n+1, excluding (0,0), (0,n+1), (n+1,0), (n+1,n+1).
        Interior nodes get indices 0 to n^2-1.
        Boundary nodes get indices n^2 to n^2+4n-1 in cyclic order.
        """
        # Step 1: Create interior nodes (i,j for i,j=1,...,n)
        current_index = 0
        for i in range(1, self.n + 1):
            for j in range(1, self.n + 1):
                self.nodes[(i, j)] = Node(i, j, self.n, current_index, is_boundary=False)
                self.index_to_coords[current_index] = (i, j)
                current_index += 1

        # Step 2: Create boundary nodes in cyclic order (counterclockwise starting at (0,1))
        # Total boundary nodes = 4n (top, right, bottom, left, each n nodes)
        boundary_start = current_index  # n^2
        # Top: i=0, j=1,...,n
        for j in range(1, self.n + 1):
            self.nodes[(0, j)] = Node(0, j, self.n, current_index, is_boundary=True)
            self.index_to_coords[current_index] = (0, j)
            current_index += 1
        # Right: i=1,...,n, j=n+1
        for i in range(1, self.n + 1):
            self.nodes[(i, self.n + 1)] = Node(i, self.n + 1, self.n, current_index, is_boundary=True)
            self.index_to_coords[current_index] = (i, self.n + 1)
            current_index += 1
        # Bottom: i=n+1, j=n,...,1
        for j in range(self.n, 0, -1):
            self.nodes[(self.n + 1, j)] = Node(self.n + 1, j, self.n, current_index, is_boundary=True)
            self.index_to_coords[current_index] = (self.n + 1, j)
            current_index += 1
        # Left: i=n,...,1, j=0
        for i in range(self.n, 0, -1):
            self.nodes[(i, 0)] = Node(i, 0, self.n, current_index, is_boundary=True)
            self.index_to_coords[current_index] = (i, 0)
            current_index += 1

    def assign_neighbors(self):
        """
        Assign neighbors to each node based on the rectangular network structure.
        Corner nodes are excluded, so boundary nodes have appropriate neighbors.
        """
        for (i, j), node in self.nodes.items():
            if node.is_boundary:
                # Boundary nodes connect to one interior neighbor
                if i == 0 and 1 <= j <= self.n:  # Top
                    neighbor_index = self.get_node_index(i + 1, j)
                    if neighbor_index is not None:
                        node.add_neighbor(neighbor_index)
                elif i == self.n + 1 and 1 <= j <= self.n:  # Bottom
                    neighbor_index = self.get_node_index(i - 1, j)
                    if neighbor_index is not None:
                        node.add_neighbor(neighbor_index)
                elif j == 0 and 1 <= i <= self.n:  # Left
                    neighbor_index = self.get_node_index(i, j + 1)
                    if neighbor_index is not None:
                        node.add_neighbor(neighbor_index)
                elif j == self.n + 1 and 1 <= i <= self.n:  # Right
                    neighbor_index = self.get_node_index(i, j - 1)
                    if neighbor_index is not None:
                        node.add_neighbor(neighbor_index)
            else:
                # Interior nodes connect to up to four neighbors
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    # Skip if neighbor is a corner node
                    if (ni, nj) in [(0, 0), (0, self.n + 1), (self.n + 1, 0), (self.n + 1, self.n + 1)]:
                        continue
                    if 0 <= ni <= self.n + 1 and 0 <= nj <= self.n + 1:
                        neighbor_index = self.get_node_index(ni, nj)
                        if neighbor_index is not None:
                            node.add_neighbor(neighbor_index)
            node.neighbors = sorted(node.neighbors)
            

    def get_node(self, i, j):
        """
        Retrieve a node by its coordinates.
        
        Args:
            i (int): x-coordinate.
            j (int): y-coordinate.
            
        Returns:
            Node: The node at (i, j), or None if invalid or a corner node.
        """
        if (i, j) in [(0, 0), (0, self.n + 1), (self.n + 1, 0), (self.n + 1, self.n + 1)]:
            return None
        return self.nodes.get((i, j))

    def get_node_index(self, i, j):
        """
        Get the index of a node given its (i, j) coordinates.
        
        Args:
            i (int): x-coordinate.
            j (int): y-coordinate.
            
        Returns:
            int: Index of the node, or None if invalid or a corner node.
        """
        if (i, j) in [(0, 0), (0, self.n + 1), (self.n + 1, 0), (self.n + 1, self.n + 1)]:
            return None
        if 0 <= i <= self.n + 1 and 0 <= j <= self.n + 1:
            # Interior nodes: i,j = 1,...,n
            if 1 <= i <= self.n and 1 <= j <= self.n:
                return (i - 1) * self.n + (j - 1)
            # Boundary nodes
            else:
                # Top: i=0, j=1,...,n
                if i == 0 and 1 <= j <= self.n:
                    return self.n * self.n + (j - 1)
                # Right: i=1,...,n, j=n+1
                elif j == self.n + 1 and 1 <= i <= self.n:
                    return self.n * self.n + self.n + (i - 1)
                # Bottom: i=n+1, j=n,...,1
                elif i == self.n + 1 and 1 <= j <= self.n:
                    return self.n * self.n + 2 * self.n + (self.n - j)
                # Left: i=n,...,1, j=0
                elif j == 0 and 1 <= i <= self.n:
                    return self.n * self.n + 3 * self.n + (self.n - i)
        return None

    def get_node_by_index(self, index):
        """
        Retrieve a node by its index.
        
        Args:
            index (int): Node index.
            
        Returns:
            Node: The node with the given index, or None if invalid.
        """
        coords = self.index_to_coords.get(index)
        return self.nodes.get(coords) if coords else None

    def get_boundary_index(self):
        """
        Update the boundary_index list with the indices of the boundary nodes.
        Excludes corner nodes.
        """
        for (i, j), node in self.nodes.items():
            if node.is_boundary:
                self.boundary_index.append(self.get_node_index(i, j))

    def get_internal_index(self):
        """
        Update the internal_index list with the indices of the internal nodes.
        """
        for (i,j),node in self.nodes.items():
            if node.is_interior:
                self.interior_index.append(self.get_node_index(i,j))

    def create_edges(self):
        """
        Create the list of edges and initialize conductivities to 0.
        Edges are stored as (x, y) where x <= y are node indices.
        """
        self.edges = []
        edge_set = set()
        
        for (i, j), node in self.nodes.items():
            node_idx = node.index
            for neighbor_idx in node.neighbors:
                edge = tuple(sorted([node_idx, neighbor_idx]))
                edge_set.add(edge)
        
        self.edges = sorted(list(edge_set))
        for edge in self.edges:
            self.conductivities[edge] = 0.0

    def get_edge(self, x, y1):
        """
        Retrieve the edge given two node indices.
        
        Args:
            x (int): Index of first node.
            y1 (int): Index of second node.
            
        Returns:
            tuple: The edge tuple (x,y1) or (y1,x) if it exists, else None.
        """
        edge = tuple(sorted([x, y1]))
        return edge if edge in self.edges else None

    def set_conductivity_by_nodes(self, x, y, gamma):
        """
        Set the conductivity for an edge.
        
        Args:
            x (int): Index of first node.
            y (int): Index of second node.
            gamma: Conductivity value.
        """
        if gamma <= 0:
            raise ValueError(f"Conductivity must be positive, got {gamma}.")
        edge = tuple(sorted([x, y]))
        if edge in self.edges:
            self.conductivities[edge] = gamma
        else:
            raise ValueError(f"Edge {edge} not found in the network.")

    def generate_conductivity(self):
        """
        Generate conductivities for edges using a formula adapted from circular_grid.py.
        For horizontal edges (same i), use sin^2 based on j.
        For vertical edges (same j), use sin^2 based on i.
        """
        for edge in self.edges:
            p = self.get_node_by_index(edge[0])
            q = self.get_node_by_index(edge[1])
            p_i, p_j = p.i, p.j
            q_i, q_j = q.i, q.j
            # self.set_conductivity_by_nodes(edge[0], edge[1], torch.abs(torch.rand(1)))
            
            if p_i == q_i:  # Horizontal edge
                gamma = (2*pow(np.sin((3*p_i)/11 + (8*2*np.pi*q_j)/(9*self.n)),2)+1)
                # gamma = (2 * (np.sin(0.2*q_i + 0.4*q_j)**2) + 1)/3    #horizontal
            else:  # Vertical edge
                gamma = (2*pow(np.sin((8*p_i)/9 + (3*2*np.pi*q_j)/(11*self.n)),2)+1)
                #gamma = (2 * (np.sin(5*q_i/9 + 4*p_j/11)**2) + 1)/3
            self.set_conductivity_by_nodes(edge[0], edge[1], gamma)

    def solve_forward_problem(self, dirichlet_data):
        """
        Solve the forward problem to compute potentials and Neumann data.
        
        Args:
            dirichlet_data (dict): Maps boundary node indices to prescribed voltages.
        
        Returns:
            tuple: (potentials, neumann_data)
                - potentials: Array of potentials at all nodes.
                - neumann_data: Dict mapping boundary node indices to currents.
        """
        total_nodes = len(self.nodes)  # n^2 + 4n
        interior_nodes = self.n * self.n
        boundary_indices = self.boundary_index
        
        if set(dirichlet_data.keys()) != set(boundary_indices):
            raise ValueError("Dirichlet data must specify voltages for all boundary nodes.")
        
        A = lil_matrix((interior_nodes, interior_nodes))
        b = np.zeros(interior_nodes)
        
        # Map interior node indices to matrix indices (0 to n^2-1)
        interior_map = {}
        for idx in range(self.n * self.n):
            interior_map[idx] = idx

        # print("interior map = ", interior_map)
        for i in range(1, self.n + 1):
            for j in range(1, self.n + 1):
                node_idx = self.get_node_index(i, j)
                node = self.get_node(i, j)
                matrix_idx = interior_map[node_idx]
                
                sum_gamma = 0.0
                for neighbor_idx in node.neighbors:
                    edge = tuple(sorted([node_idx, neighbor_idx]))
                    if edge not in self.conductivities:
                        raise ValueError(f"Edge {edge} conductivity not found.")
                    gamma = self.conductivities[edge]
                    if gamma <= 0:
                        raise ValueError(f"Conductivity for edge {edge} must be positive.")
                    
                    sum_gamma += gamma
                    if neighbor_idx in boundary_indices:
                        b[matrix_idx] += gamma * dirichlet_data[neighbor_idx]
                    else:
                        A[matrix_idx, interior_map[neighbor_idx]] = -gamma
                
                A[matrix_idx, matrix_idx] = sum_gamma

        # print("in lil matrix form, A = ", A)
        
        A = A.tocsr()
        # print("in csr matrix form, A  = ",A)
        try:
            interior_potentials = spsolve(A, b)
        except Exception as e:
            raise RuntimeError(f"Failed to solve linear system: {e}")
        
        potentials = np.zeros(total_nodes)
        #potentials gives the potentials for all the nodes in the grid
        for i in range(1, self.n + 1):
            for j in range(1, self.n + 1):
                node_idx = self.get_node_index(i, j)
                potentials[node_idx] = interior_potentials[interior_map[node_idx]]
        for idx in boundary_indices:
            potentials[idx] = dirichlet_data[idx]
        
        neumann_data= {}      #resulting current at the boundaries
        neumann_data_int = {}    #resulting current at the internal nodes but this wont be used
        for idx in boundary_indices:
            node = self.get_node_by_index(idx)
            if len(node.neighbors) != 1:
                raise ValueError(f"Boundary node {idx} should have exactly one neighbor.")
            neighbor_idx = node.neighbors[0]
            edge = tuple(sorted([idx, neighbor_idx]))
            gamma = self.conductivities[edge]
            neumann_data[idx] = gamma * (potentials[idx] - potentials[neighbor_idx])

        internal_indices = self.interior_index
        for idx in internal_indices:
            node = self.get_node_by_index(idx)
            neumann_data_int[idx] = 0
            if len(node.neighbors)!=4:
                raise ValueError(f"Internal node {idx} should have exactly four neighbor.")
            neighbors = node.neighbors
            for i in neighbors:
                edge = tuple(sorted([idx,i]))
                gamma = self.conductivities[edge]
                neumann_data_int[idx]+= gamma*(potentials[idx]- potentials[i])
                
            
            
        
        return potentials, neumann_data #, neumann_data_int
        

    def get_forward_problem_lin_solver(self,dirichlet_data):
        """
        Solve the forward problem to compute potentials and Neumann data.
        
        Args:
            dirichlet_data (dict): Maps boundary node indices to prescribed voltages.
        
        Returns:
            only the matrix A,b in Ax = b for the forward problem
        """
        total_nodes = len(self.nodes)  # n^2 + 4n
        interior_nodes = self.n * self.n
        boundary_indices = self.boundary_index
        
        if set(dirichlet_data.keys()) != set(boundary_indices):
            raise ValueError("Dirichlet data must specify voltages for all boundary nodes.")
        
        A = np.zeros((interior_nodes, interior_nodes))
        b = np.zeros(interior_nodes)
        
        # Map interior node indices to matrix indices (0 to n^2-1)
        interior_map = {}
        for idx in range(self.n * self.n):
            interior_map[idx] = idx

        # print("interior map = ", interior_map)
        for i in range(1, self.n + 1):
            for j in range(1, self.n + 1):
                node_idx = self.get_node_index(i, j)
                node = self.get_node(i, j)
                matrix_idx = interior_map[node_idx]
                
                sum_gamma = 0.0
                for neighbor_idx in node.neighbors:
                    edge = tuple(sorted([node_idx, neighbor_idx]))
                    if edge not in self.conductivities:
                        raise ValueError(f"Edge {edge} conductivity not found.")
                    gamma = self.conductivities[edge]
                    if gamma <= 0:
                        raise ValueError(f"Conductivity for edge {edge} must be positive.")
                    
                    sum_gamma += gamma
                    if neighbor_idx in boundary_indices:
                        b[matrix_idx] += gamma * dirichlet_data[neighbor_idx]
                    else:
                        A[matrix_idx, interior_map[neighbor_idx]] = -gamma
                
                A[matrix_idx, matrix_idx] = sum_gamma

        try:
            interior_potentials = sl.linear_solver(A, b)
        except Exception as e:
            raise RuntimeError(f"Failed to solve linear system: {e}")
            
        potentials = np.zeros(total_nodes)
        #potentials gives the potentials for all the nodes in the grid
        for i in range(1, self.n + 1):
            for j in range(1, self.n + 1):
                node_idx = self.get_node_index(i, j)
                potentials[node_idx] = interior_potentials[interior_map[node_idx]]
        for idx in boundary_indices:
            potentials[idx] = dirichlet_data[idx]
        
        neumann_data= {}
        neumann_data_int = {}
        for idx in boundary_indices:
            node = self.get_node_by_index(idx)
            if len(node.neighbors) != 1:
                raise ValueError(f"Boundary node {idx} should have exactly one neighbor.")
            neighbor_idx = node.neighbors[0]
            edge = tuple(sorted([idx, neighbor_idx]))
            gamma = self.conductivities[edge]
            neumann_data[idx] = gamma * (potentials[idx] - potentials[neighbor_idx])

        internal_indices = self.interior_index
        for idx in internal_indices:
            node = self.get_node_by_index(idx)
            neumann_data_int[idx] = 0
            if len(node.neighbors)!=4:
                raise ValueError(f"Internal node {idx} should have exactly four neighbor.")
            neighbors = node.neighbors
            for i in neighbors:
                edge = tuple(sorted([idx,i]))
                gamma = self.conductivities[edge]
                neumann_data_int[idx]+= gamma*(potentials[idx]- potentials[i])
        
        
        return potentials, neumann_data, neumann_data_int
        

    def visualize_network(self):
        """
        Visualize the rectangular network using Matplotlib.
        Nodes are plotted with coordinates, edges as lines.
        Boundary nodes are blue, interior nodes are purple.
        Edge colors reflect conductivity range.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot nodes
        for (i, j), node in self.nodes.items():
            color = 'blue' if node.is_boundary else 'purple'
            plt.scatter(node.x, node.y, c=[color], s=50)
            plt.text(node.x + 0.1, node.y + 0.1, f"{node.index}", fontsize=8)
    
        # Get min and max conductivities
        values = list(self.conductivities.values())
        vmin = min(values) if values else 0.0
        vmax = max(values) if values else 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot edges
        for edge in self.edges:
            node1 = self.get_node_by_index(edge[0])
            node2 = self.get_node_by_index(edge[1])
            conductivity = self.conductivities.get(edge, 0.0)
            plt.plot([node1.x, node2.x], [node1.y, node2.y],
                     color=viridis(norm(conductivity)), linewidth=1.5)
    
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=viridis, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Conductivity')
    
        # Plot settings
        ax.set_aspect('equal')
        ax.set_title(f"Rectangular Network ({self.n}x{self.n})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        plt.show()

    def __repr__(self):
        return f"GridStructure(n={self.n}, nodes={len(self.nodes)})"