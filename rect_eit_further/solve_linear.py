import numpy as np
def forward_substitution(L, b):
    """
    To solve the equations of the form Lx = b where L is a lower traingular matrix
    """
    x = np.zeros(np.array(b).size)
    for i in range(len(x)):
        sum = np.sum([L[i][j]*x[j] for j in range(i)])
        x[i] = (b[i]-sum)/L[i][i]
    return x

def backward_substitution(U, b):
    """
    To solve the equations of the form Ux = b where U is an upper traingular matrix


    """
    x = np.zeros(np.array(b).size)
    for i in range(len(x)-1,-1,-1):
        sum = np.sum([U[i][j]*x[j] for j in range(i,len(x))])
        x[i] = (b[i]-sum)/U[i][i]
    return x

def gaussian_from_coeffs(coeffs, dim):

    """
    a function outputing the Gaussian transformation matrix $\mathbf{M}_k$ associated to a series of coefficients $\mathbf{c}^{(k)}$.


    """
    I = np.zeros((dim,dim))
    for i in range(dim):
        I[i][i] = 1
    # print(I)
    #I is the identity matrix
    j=-1
    for i in range(len(coeffs)):
        if(coeffs[i]!=0):
            j = i
            break
    e_i = np.zeros((1,dim))
    e_i[0][j-1]=1
    # print(e_i)
    coeffs = coeffs.reshape(dim,1)
    return I-coeffs@e_i


def compute_coeff(A,i):    #compute c_(i), starts from 0,1,....N
    c = np.zeros(A.shape[0])
    for j in range(i+1,A.shape[0]):
        c[j] = A[j][i]/A[i][i]
    return c

def lu_decomposition(A):
    """
    to compute the lu decomposition of the matrix A

    """
    B=A
    dim = A.shape[0]
    I = np.zeros((dim,dim))
    for i in range(dim):
        I[i][i] = 1
    M_i_evolved = I
    for i in range(dim):
        c_i = compute_coeff(B,i)
        M_i = gaussian_from_coeffs(c_i,dim)
        M_i_evolved = M_i@M_i_evolved
        B = M_i@B
    U = B
    L = np.linalg.inv(M_i_evolved)
        
    return L, U

def linear_solver(A,b):

    """
    Solves the system of the form Ax = b using LU decompositon and returns x

    """
    L,U = lu_decomposition(A)
    y = forward_substitution(L,b)
    x = backward_substitution(U,y)
    return x