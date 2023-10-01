from functools import wraps
from time import time
import psutil

import numpy as np
import torch

def timing_and_memory(f):
    @wraps(f)
    def wrap(*args, **kw):
        # Initial time
        ts = time()
        
        # Initial memory
        process = psutil.Process()
        init_cpu_memory = process.memory_info().rss
        
        if torch.cuda.is_available():
            init_cuda_allocated = torch.cuda.memory_allocated()
            init_cuda_reserved = torch.cuda.memory_reserved()
        else:
            init_cuda_allocated = 0
            init_cuda_reserved = 0
        
        # Call the wrapped function
        result = f(*args, **kw)
        
        # Final time
        te = time()
        
        # Final memory
        final_cpu_memory = process.memory_info().rss
        
        if torch.cuda.is_available():
            final_cuda_allocated = torch.cuda.memory_allocated()
            final_cuda_reserved = torch.cuda.memory_reserved()
            
            print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
            print('CPU Memory Used increased by: %d bytes' % (final_cpu_memory - init_cpu_memory))
            print('CUDA Memory Allocated increased by: %d bytes' % (final_cuda_allocated - init_cuda_allocated))
            print('CUDA Memory Reserved increased by: %d bytes' % (final_cuda_reserved - init_cuda_reserved))
        else:
            print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
            print('CPU Memory Used increased by: %d MB' % ((final_cpu_memory - init_cpu_memory) / (1000 * 1000)))
            print('CUDA is not available, measured only CPU memory usage')
        
        return result
    return wrap

def main(C, L_x, L_y, d=5):
# Convert the numpy matrices to PyTorch tensors
    C_torch = torch.tensor(C, dtype=torch.float64)
    L_x_torch = torch.tensor(L_x, dtype=torch.float64)
    L_y_torch = torch.tensor(L_y, dtype=torch.float64)

# Compute the matrix difference CLx - LyC and calculate the Frobenius norm
    explicit = C @ L_x - L_y @ C

# Compute the equivalent norm using the vectorized form with Kronecker products
    vec_C = C.flatten('F')
    left = np.kron(L_x, np.eye(d))
    right = np.kron(np.eye(d), L_y)
    expanded = (left - right) @ vec_C

    print("numpy version: ", np.linalg.norm(np.abs(explicit.flatten('F') - expanded)))

# Compute the matrix difference CLx - LyC and calculate the Frobenius norm using built-in function in PyTorch
    explicit_torch = C_torch @ L_x_torch - L_y_torch @ C_torch


# Compute the equivalent norm using the vectorized form with built-in Kronecker product in PyTorch
    vec_C_torch = C_torch.T.flatten()  # Equivalent to flatten in NumPy
    left_torch = torch.kron(L_x_torch, torch.eye(d, dtype=torch.float64))
    right_torch = torch.kron(torch.eye(d, dtype=torch.float64), L_y_torch)
    expanded_torch = (left_torch - right_torch) @ vec_C_torch

    assert np.allclose(vec_C, vec_C_torch.numpy())
    assert np.allclose(left, left_torch.numpy())
    assert np.allclose(right, right_torch.numpy())
    assert np.allclose(explicit, explicit_torch.numpy())
    assert np.allclose(expanded, expanded_torch.numpy())

    print("torch version:", torch.norm(torch.abs(explicit_torch.T.flatten() - expanded_torch)))

def verify_gradient(d=5):

# Define dimensions and generate random matrices
    m = 10
    d = 5
    A = np.random.rand(d, m)
    B = np.random.rand(d, m)
    C = np.random.rand(d, d)
    L_x = np.random.rand(d, d)
    L_y = np.random.rand(d, d)
    vec_B = B.flatten('F')
    vec_C = C.flatten('F')
    lmdba = 0.1

    At_Ik = np.kron(A.T, np.eye(d))
# Compute Delta
    Delta = np.kron(L_x, np.eye(d)) - np.kron(np.eye(d), L_y)

# Compute the energy function at vec_C
    def cost(vec_C):
        return np.linalg.norm(At_Ik @ vec_C - vec_B)**2 + lmdba * np.linalg.norm(Delta @ vec_C) ** 2

    E_vec_C = cost(vec_C)

# Define a small number for finite difference approximation
    h = 1e-5

# Compute the analytical gradient
    analytical_gradient = ((At_Ik.T @ At_Ik + lmdba * (Delta.T @ Delta)) @ vec_C - At_Ik.T @ vec_B)

# Initialize a vector to store the numerical gradient
    numerical_gradient = np.zeros_like(vec_C)

# Compute the numerical gradient using finite difference approximation for each component of vec_C
    for i in range(len(vec_C)):
        vec_C_perturbed = vec_C.copy()
        vec_C_perturbed[i] += h
        E_vec_C_perturbed = cost(vec_C_perturbed)
        numerical_gradient[i] = (E_vec_C_perturbed - E_vec_C) / h

# Compute the norm of the difference between the analytical and numerical gradients
    gradient_error_norm = np.linalg.norm(analytical_gradient - numerical_gradient)

    print("Analytical Gradient:", analytical_gradient)
    print("Numerical Gradient:", numerical_gradient)
    print("Gradient Error Norm:", gradient_error_norm)


def sparse_kron(D1, D2):
    # Assuming D1 and D2 are diagonal matrices represented as 1D tensors
    i = torch.cartesian_prod(torch.arange(len(D1)), torch.arange(len(D2))).t()
    v = torch.ger(D1, D2).flatten()
    size = (len(D1) * len(D2), len(D1) * len(D2))
    return torch.sparse_coo_tensor(i, v, size, dtype=D1.dtype, device=D1.device)


def sparse_kron_A_I(dense_matrix, identity_size):
    m, n = dense_matrix.shape
    sparse_shape = (m * identity_size, n * identity_size)
    
    # Find the non-zero entries in the dense matrix
    rows, cols = torch.nonzero(dense_matrix, as_tuple=True)
    
    # Calculate the starting row and column indices in the sparse matrix for each non-zero entry in the dense matrix
    row_starts = rows * identity_size
    col_starts = cols * identity_size
    
    # Generate the row and column indices within each block for the sparse matrix
    block_indices = torch.arange(identity_size)
    
    # Calculate the indices for the non-zero entries in the sparse matrix (only diagonal blocks)
    sparse_rows = (row_starts[:, None] + block_indices).reshape(-1)
    sparse_cols = (col_starts[:, None] + block_indices).reshape(-1)
    
    # Generate the values for the non-zero entries in the sparse matrix
    sparse_values = dense_matrix[rows, cols].repeat_interleave(identity_size)
    
    # Create the sparse matrix
    sparse_indices = torch.stack((sparse_rows, sparse_cols))
    sparse_matrix = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_shape)
    
    return sparse_matrix


@timing_and_memory
def test_expanded(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3):
    A = evecs_trans_x @ feat_x
    B = evecs_trans_y @ feat_y
    # A and B should be same shape
    k, m = A.size(0), A.size(1)

    vec_B = B.T.reshape(m * k, 1).contiguous()

    A_t = A.T.contiguous()
    Ik = torch.eye(k, device=A.device, dtype=torch.float32)

    At_Ik = torch.kron(A_t, Ik)

    lx = torch.diag(evals_x.squeeze(0))
    ly = torch.diag(evals_y.squeeze(0))
    lx_Ik = torch.kron(lx, Ik)
    Ik_ly = torch.kron(Ik, ly)
    Delta = (lx_Ik - Ik_ly)

    first = At_Ik.T @ At_Ik
    second = Delta.T @ Delta
    rhs = At_Ik.T @ vec_B
    op = first + lambda_param * second

    C = torch.linalg.solve(op, rhs)

    return C.reshape(k, k).T

@timing_and_memory
def test_memory_efficient(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3):
    A = evecs_trans_x @ feat_x
    B = evecs_trans_y @ feat_y
    # A and B should be same shape
    k, m = A.size(0), A.size(1)

    vec_B = B.T.reshape(m * k, 1).contiguous()

    A_t = A.T.contiguous()

    At_Ik = sparse_kron_A_I(A_t, k)
    lx_Ik = sparse_kron(evals_x.squeeze(0), torch.ones(k, device=A.device, dtype=A.dtype))
    Ik_ly = sparse_kron(torch.ones(k, device=A.device, dtype=A.dtype), evals_y.squeeze(0))

    Delta = (lx_Ik - Ik_ly)

    first = At_Ik.T @ At_Ik
    second = Delta.T @ Delta
    rhs = At_Ik.T @ vec_B
    op = first + lambda_param * second

    C = torch.linalg.solve(op, rhs)

    return C.reshape(k, k).T


if __name__ == "__main__":
    N = 20
    C = np.random.rand(N, N)
    # L_x = np.diag(np.random.rand(N))
    # L_y = np.diag(np.random.rand(N))
    # main(C, L_x, L_y, d=N)
    # verify_gradient(d=N)

    # generate random torch data for testing as input args
    N = 5000
    m = 128
    k = 30
    feats_x = torch.rand(N, m)
    feats_y = torch.rand(N, m)
    evals_x = torch.rand(k)
    evals_y = torch.rand(k)
    evecs_trans_x = torch.rand(k, N)
    evecs_trans_y = torch.rand(k, N)
    test_expanded(feats_x, feats_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
    test_memory_efficient(feats_x, feats_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
