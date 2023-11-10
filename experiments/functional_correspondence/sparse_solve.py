from kron import main
import torch

# PyTorch also doesn't support any sparse solver
# -> workaround by using cupy with a custom backward pass
# https://docs.cupy.dev/en/stable/reference/scipy_sparse_linalg.html#solving-linear-problems
# https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html
# Note that the default cupy version clashes with the default pytorch version
# in colab -> use cupy 10
import cupy as cp
print('CuPy version:',cp.__version__)
print('Running on ',cp.array([1]).data.device)

# Convenience function to map a torch COO tensor in a cupy one
def coo_torch2cupy(A):
  A = A.data.coalesce()
  Avals_cp = cp.asarray(A.values())
  Aidx_cp = cp.asarray(A.indices())
  return cp.sparse.coo_matrix((Avals_cp, Aidx_cp))

# Custom PyTorch sparse solver exploiting a CuPy backend
# See https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html
class SparseSolve(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, b):
    # Sanity check
        if A.ndim != 2 or (A.shape[0] < A.shape[1]):
            raise ValueError("A should be a square 2D matrix or overdetermined.")
    # Transfer data to CuPy
        A_cp = coo_torch2cupy(A)
        b_cp = cp.asarray(b.data)
    # Solver the sparse system
        # Make use of a factorisation (only the solver is then on the GPU)
        x_cp = lsqr(A_cp, b_cp)[0]
    # Transfer (dense) result back to PyTorch
        x = torch.as_tensor(x_cp, device=A.device)
    # Save context for backward pass
        ctx.save_for_backward(A, b, x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        A, _, x = ctx.saved_tensors
        
        # Calculate the gradient with respect to x
        grad_x = A.t() @ (A @ x - b)  # Adjusted for least squares
        
        # Backpropagate the gradient to previous layer
        return None, grad_x * grad_output

def lsqr(A, b):
    """Solves linear system with QR decomposition.

    Find the solution to a large, sparse, linear system of equations.
    The function solves ``Ax = b``. Given two-dimensional matrix ``A`` is
    decomposed into ``Q * R``.

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.csr_matrix): The input matrix
            with dimension ``(N, N)``
        b (cupy.ndarray): Right-hand side vector.

    Returns:
        tuple:
            Its length must be ten. It has same type elements
            as SciPy. Only the first element, the solution vector ``x``, is
            available and other elements are expressed as ``None`` because
            the implementation of cuSOLVER is different from the one of SciPy.
            You can easily calculate the fourth element by ``norm(b - Ax)``
            and the ninth element by ``norm(x)``.

    .. seealso:: :func:`scipy.sparse.linalg.lsqr`
    """
    import numpy
    import cupy
    from cupyx.scipy import sparse
    from cupy.cuda import cusolver
    from cupy.cuda import device
    from cupy.cuda import runtime
    from cupyx import cusparse
    from cupy.linalg import _util
    if runtime.is_hip:
        raise RuntimeError('HIP does not support lsqr')
    if not sparse.isspmatrix_csr(A):
        A = sparse.csr_matrix(A)
    # csr_matrix is 2d
    _util._assert_stacked_square(A)
    _util._assert_cupy_array(b)
    m = A.shape[0]
    if b.ndim != 1 or len(b) != m:
        raise ValueError('b must be 1-d array whose size is same as A')

    # Cast to float32 or float64
    if A.dtype == 'f' or A.dtype == 'd':
        dtype = A.dtype
    else:
        dtype = numpy.promote_types(A.dtype, 'f')

    handle = device.get_cusolver_sp_handle()
    nnz = A.nnz
    tol = 1.0
    reorder = 1
    x = cupy.empty(m, dtype=dtype)
    singularity = numpy.empty(1, numpy.int32)

    if dtype == 'f':
        csrlsvqr = cusolver.scsrlsvqr
    else:
        csrlsvqr = cusolver.dcsrlsvqr
    csrlsvqr(
        handle, m, nnz, A._descr.descriptor, A.data.data.ptr,
        A.indptr.data.ptr, A.indices.data.ptr, b.data.ptr, tol, reorder,
        x.data.ptr, singularity.ctypes.data)

    # The return type of SciPy is always float64. Therefore, x must be casted.
    # NOTE: we comment this line out from the original cupy code
    # x = x.astype(numpy.float64)
    ret = (x, None, None, None, None, None, None, None, None, None)
    return ret


if __name__ == "__main__":
    print('PyTorch version:',torch.__version__)

    torchdevice = torch.device('cpu')
    if torch.cuda.is_available():
        torchdevice = torch.device('cuda')
    print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
    print('Running on ' + str(torchdevice))

    sparsesolve = SparseSolve.apply

# Test matrix-vector solver
    Aref = torch.randn(3,3, dtype=torch.float64, requires_grad=False, device=torchdevice).to_sparse()
    Aref.requires_grad=False
    bref = torch.randn(3, dtype=torch.float64, requires_grad=False, device=torchdevice)

    A = Aref.detach().clone().requires_grad_(True)
    b = bref.detach().clone().requires_grad_(True)

# Solve
    x = sparsesolve(A,b)

# random scalar function to mimick a loss
    loss = x.sum()
    loss.backward()

    print('x',x)
    with torch.no_grad(): print('allclose:',torch.allclose(A @ x, b))
    print('A.grad',A.grad)
    print('b.grad',b.grad)

# Compare with dense op
    A = Aref.detach().clone().to_dense().requires_grad_(True)
    b = bref.detach().clone().requires_grad_(True)
    x = torch.linalg.solve(A,b)
    loss = x.sum()
    loss.backward()
    print('x',x)
    with torch.no_grad(): print('allclose:',torch.allclose(A @ x, b))
    print('A.grad',A.grad)
    print('b.grad',b.grad)

# Test matrix-matrix solver
    Aref = torch.randn(3,3, dtype=torch.float64, requires_grad=False, device=torchdevice).to_sparse()
    Aref.requires_grad=False
    bref = torch.randn(3,2, dtype=torch.float64, requires_grad=False, device=torchdevice)

    A = Aref.detach().clone().requires_grad_(True)
    b = bref.detach().clone().requires_grad_(True)

# Solve
    x = sparsesolve(A,b)

# random scalar function to mimick a loss
    loss = x.sum()
    loss.backward()

    print('x',x)
    with torch.no_grad(): print('allclose:',torch.allclose(A @ x, b))
    print('A.grad',A.grad)
    print('b.grad',b.grad)

# Compare with dense op
    A = Aref.detach().clone().to_dense().requires_grad_(True)
    b = bref.detach().clone().requires_grad_(True)
    x = torch.linalg.solve(A,b)
    loss = x.sum()
    loss.backward()
    print('x',x)
    print('allclose:',torch.allclose(A @ x, b))
    print('A.grad',A.grad)
    print('b.grad',b.grad)

# Now try some gradcheck
    A = torch.randn(3,3, dtype=torch.float64, device=torchdevice).to_sparse()
    A.requires_grad=True
    b = torch.randn(3, dtype=torch.float64, requires_grad=True, device=torchdevice)
    torch.autograd.gradcheck(sparsesolve, [A, b], check_sparse_nnz=True, raise_exception=True)
