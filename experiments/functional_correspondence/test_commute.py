import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))
import diffusion_net


def test_expanded(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3):
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

    At_Ik = torch.kron(A_t, torch.eye(k, device=A.device))

    lx_Ik = torch.kron(torch.diag(evals_x.squeeze(0)), torch.eye(k, device=A.device))
    Ik_ly = torch.kron(torch.eye(k, device=A.device), torch.diag(evals_y.squeeze(0)))
    Delta = (lx_Ik - Ik_ly)

    first = At_Ik.T @ At_Ik
    second = Delta.T @ Delta
    rhs = At_Ik.T @ vec_B
    op = first + lambda_param * second

    C = torch.linalg.solve(op.to_dense(), rhs)

    return C.reshape(k, k).T


def test_explicit(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3):
    """
    Computes the functional map correspondence matrix C given features from two shapes.

    Has no trainable parameters.
    """

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


if __name__ == "__main__":
    # generate random torch data for testing as input args
    device = "cuda:1"
    # feats_x = torch.rand(N, m, device=device)
    # feats_y = torch.rand(N, m, device=device)
    # evals_x = torch.rand(k, device=device)
    # evals_y = torch.rand(k, device=device)
    # evecs_trans_x = torch.rand(k, N, device=device)
    # evecs_trans_y = torch.rand(k, N, device=device)
    # save all input data with torch
    loaded_params = torch.load('input_params.pth')

    # Extracting each parameter
    feat_x = loaded_params['feat_x'].to(device)
    feat_y = loaded_params['feat_y'].to(device)
    evals_x = loaded_params['evals_x'].to(device)
    evals_y = loaded_params['evals_y'].to(device)
    evecs_trans_x = loaded_params['evecs_trans_x'].to(device)
    evecs_trans_y = loaded_params['evecs_trans_y'].to(device)
    for i in np.linspace(0, 100, 20):
        C_expanded = test_expanded(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=i)
        C_explicit = test_explicit(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=i).squeeze(0)
        print((C_expanded.real - C_explicit).norm())
