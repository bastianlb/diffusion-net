import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))
import diffusion_net


def compute_correspondence_iterative(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3):
    """
    Computes the functional map correspondence matrix C given features from two shapes.

    Has no trainable parameters.
    """
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer

    feat_x, feat_y = feat_x.unsqueeze(0), feat_y.unsqueeze(0)
    evecs_trans_x, evecs_trans_y = evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0)
    evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)

    F_hat = torch.bmm(evecs_trans_x, feat_x)
    G_hat = torch.bmm(evecs_trans_y, feat_y)
    A, B = F_hat.squeeze(0), G_hat.squeeze(0)

    C = cp.Variable((evals_x.size(1), evals_y.size(1)))
    A_var = cp.Parameter((A.size(0), A.size(1)))
    B_var = cp.Parameter((B.size(0), B.size(1)))
    L1_var = cp.Parameter((evals_x.size(1)))
    L2_var = cp.Parameter((evals_y.size(1)))
    objective = cp.Minimize(cp.norm(C @ A_var - B_var, 'fro') +
                            lambda_param * cp.norm(C @ cp.diag(L1_var) - cp.diag(L2_var) @ C, 'fro'))

    problem = cp.Problem(objective)

    cvxpylayer = CvxpyLayer(problem, parameters=[A_var, B_var, L1_var, L2_var],
                            variables=[C])
    C, = cvxpylayer(A, B, evals_x.squeeze(0), evals_y.squeeze(0))

    return C.to(feat_x.device).unsqueeze(0)


def compute_correspondence_explicit(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3):
    """
    Computes the functional map correspondence matrix C given features from two shapes.

    Has no trainable parameters.
    """

    # save all input data with torch
    torch.save({
        'feat_x': feat_x,
        'feat_y': feat_y,
        'evals_x': evals_x,
        'evals_y': evals_y,
        'evecs_trans_x': evecs_trans_x,
        'evecs_trans_y': evecs_trans_y,
        'lambda_param': lambda_param
    }, 'input_params.pth')


    feat_x, feat_y = feat_x.unsqueeze(0), feat_y.unsqueeze(0)
    evecs_trans_x, evecs_trans_y = evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0)
    evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)

    F_hat = torch.bmm(evecs_trans_x, feat_x)
    G_hat = torch.bmm(evecs_trans_y, feat_y)
    A, B = F_hat, G_hat

    D = torch.repeat_interleave(evals_x.unsqueeze(1), repeats=evals_x.size(1), dim=1)
    D = (D - torch.repeat_interleave(evals_y.unsqueeze(2), repeats=evals_x.size(1), dim=2)) ** 2

    A_t = A.transpose(1, 2)
    A_A_t = torch.bmm(A, A_t)
    B_A_t = torch.bmm(B, A_t)

    C_i = []
    for i in range(evals_x.size(1)):
        D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
        C = torch.bmm(torch.inverse(A_A_t + lambda_param * D_i), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2))
        C_i.append(C.transpose(1, 2))
    C = torch.cat(C_i, dim=1)

    return C


def sparse_kron(D1, D2):
    # Assuming D1 and D2 are diagonal matrices represented as 1D tensors
    i = torch.cartesian_prod(torch.arange(len(D1)), torch.arange(len(D2))).t()
    v = torch.outer(D1, D2).flatten()
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
    block_indices = torch.arange(identity_size, device=dense_matrix.device)
    
    # Calculate the indices for the non-zero entries in the sparse matrix (only diagonal blocks)
    sparse_rows = (row_starts[:, None] + block_indices).reshape(-1)
    sparse_cols = (col_starts[:, None] + block_indices).reshape(-1)
    
    # Generate the values for the non-zero entries in the sparse matrix
    sparse_values = dense_matrix[rows, cols].repeat_interleave(identity_size).to(dense_matrix.device)
    
    # Create the sparse matrix
    sparse_indices = torch.stack((sparse_rows, sparse_cols))
    sparse_matrix = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_shape, device=dense_matrix.device)
    
    return sparse_matrix


def compute_correspondence_expanded(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3):
    """
    Computes the functional map correspondence matrix C given features from two shapes.

    Has no trainable parameters.
    """
    A = evecs_trans_x @ feat_x
    B = evecs_trans_y @ feat_y
    # A and B should be same shape
    k, m = A.size(0), A.size(1)

    vec_B = B.T.reshape(m * k, 1).contiguous()

    A_t = A.T.contiguous()

    # At_Ik = sparse_kron_A_I(A_t, k)
    At_Ik = torch.kron(A_t, torch.eye(k, device=A.device))
    # assert torch.equal(At_Ik.to_dense(), At_Ik_dense)

    # lx_Ik = sparse_kron(evals_x.squeeze(0), torch.ones(k, device=A.device))
    lx_Ik = torch.kron(torch.diag(evals_x.squeeze(0)), torch.eye(k, device=A.device))
    # assert torch.equal(lx_Ik.to_dense(), lx_Ik_dense)
    # Ik_ly = sparse_kron(torch.ones(k, device=A.device), evals_y.squeeze(0))
    Ik_ly = torch.kron(torch.eye(k, device=A.device), torch.diag(evals_y.squeeze(0)))
    # assert torch.equal(Ik_ly.to_dense(), Ik_ly_dense)
    Delta = (lx_Ik - Ik_ly)

    first = At_Ik.T @ At_Ik
    second = Delta.T @ Delta
    rhs = At_Ik.T @ vec_B
    op = first + lambda_param * second

    C = torch.linalg.solve(op.to_dense(), rhs)

    return C.reshape(k, k).T


class FunctionalMapCorrespondenceWithDiffusionNetFeatures(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, n_feat=128, n_fmap=30, lambda_=1e-3, input_features="xyz", lambda_param=1e-3):
        super().__init__()

        C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

        self.feature_extractor = diffusion_net.layers.DiffusionNet(
            C_in=C_in,
            C_out=n_feat,
            C_width=128,
            N_block=4,
            dropout=True,
        )

        self.n_fmap = n_fmap
        self.input_features = input_features
        self.lambda_param = lambda_param

    def forward(self, shape1, shape2, i=0):
        verts1, faces1, frames1, mass1, L1, evals1, evecs1, gradX1, gradY1, hks1, vts1 = shape1
        verts2, faces2, frames2, mass2, L2, evals2, evecs2, gradX2, gradY2, hks2, vts2 = shape2

        # set features
        if self.input_features == "xyz":
            features1, features2 = verts1, verts2
        elif self.input_features == "hks":
            features1, features2 = hks1, hks2

        feat1 = self.feature_extractor(features1, mass1, L=L1, evals=evals1, evecs=evecs1,
                                       gradX=gradX1, gradY=gradY1, faces=faces1)
        feat2 = self.feature_extractor(features2, mass2, L=L2, evals=evals2, evecs=evecs2,
                                       gradX=gradX2, gradY=gradY2, faces=faces2)

        # construct the fmap
        evecs_trans1, evecs_trans2 = evecs1.t()[:self.n_fmap] @ torch.diag(mass1), evecs2.t()[:self.n_fmap] @ torch.diag(mass2)
        evals1, evals2 = evals1[:self.n_fmap], evals2[:self.n_fmap]
        import time
        t0 = time.time()
        C_expanded = compute_correspondence_expanded(feat1, feat2, evals1, evals2, evecs_trans1, evecs_trans2, lambda_param=self.lambda_param)
        print("expanded", time.time()-t0)
        t0 = time.time()
        C_explicit= compute_correspondence_explicit(feat1, feat2, evals1, evals2, evecs_trans1, evecs_trans2, lambda_param=self.lambda_param)
        print("explicit", time.time() - t0)
        C_diff = (torch.abs(C_expanded - C_explicit))
        __import__('ipdb').set_trace()
        print("diff: ", torch.sum(C_diff).item())
        import matplotlib.pyplot as plt

        # Create a figure with three subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plotting the first heatmap
        heatmap1 = axs[0].imshow(C_explicit.squeeze(0).detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        axs[0].set_title('C_explicit')
        fig.colorbar(heatmap1, ax=axs[0], fraction=0.046, pad=0.04)  # Adjust colorbar size and position

# Plotting the second heatmap
        heatmap2 = axs[1].imshow(C_expanded.squeeze(0).detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        axs[1].set_title('C_expanded')
        fig.colorbar(heatmap2, ax=axs[1], fraction=0.046, pad=0.04)  # Adjust colorbar size and position

# Plotting the difference heatmap
        heatmap3 = axs[2].imshow(C_diff.squeeze(0).detach().cpu().numpy(), cmap='coolwarm', interpolation='nearest')
        axs[2].set_title('Difference (C_explicit - C_expanded)')
        fig.colorbar(heatmap3, ax=axs[2], fraction=0.046, pad=0.04)  # Adjust colorbar size and position

# Save the figure
        plt.savefig(f'figs/heatmaps_and_difference_{i}.png', format='png')

        C_pred = C_explicit

        return C_pred, feat1, feat2
