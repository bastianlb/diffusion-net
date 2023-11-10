import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))


def _get_mask(evals1, evals2, resolvant_gamma):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1 / scaling_factor, evals2 / scaling_factor
    evals_gamma1 = (evals1 ** resolvant_gamma)[None, :]
    evals_gamma2 = (evals2 ** resolvant_gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()


def get_mask(evals1, evals2, resolvant_gamma=0.5):
    masks = []
    for bs in range(evals1.shape[0]):
        masks.append(_get_mask(evals1[bs], evals2[bs], resolvant_gamma))
    return torch.stack(masks, dim=0)


def test_expanded(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3):
    """
    Computes the functional map correspondence matrix C given features from two shapes.

    Has no trainable parameters.
    """
    A = evecs_trans_x @ feat_x
    B = evecs_trans_y @ feat_y
    # A and B should be same shape
    k, m = A.size(0), A.size(1)

    # instead of the mask, we need to compute the resolvant operator
    scaling_factor = max(torch.max(evals_x), torch.max(evals_y))
    complex_unit_vec = torch.tensor(1j, device=evals_x.device) * torch.ones(evals_x.shape[0], device=evals_x.device)
    Res_x = evals_x.squeeze(0) / scaling_factor ** 0.5 - complex_unit_vec
    Res_y = evals_y.squeeze(0) / scaling_factor ** 0.5 - complex_unit_vec

    vec_B = B.T.reshape(m * k, 1).contiguous()

    A_t = A.T.contiguous()

    At_Ik = torch.kron(A_t, torch.eye(k, device=A.device))

    # inverse of a diagonal matrix is just the inverse of each entry on the diagonal
    lx_Ik = torch.kron(torch.diag(1 / Res_x), torch.eye(k, device=A.device))
    Ik_ly = torch.kron(torch.eye(k, device=A.device), torch.diag(1 / Res_y))
    Delta = (lx_Ik - Ik_ly)

    first = At_Ik.T @ At_Ik
    second = Delta.T @ Delta
    # rhs needs to be converted to complex
    rhs = At_Ik.T @ vec_B + 0j
    op = first + lambda_param * second

    # we only care about the real part of the system
    C = torch.linalg.solve(op, rhs).real

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

    D = get_mask(evals_x, evals_y)

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
    C_expanded = test_expanded(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
    C_explicit = test_explicit(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y).squeeze(0)
    assert (C_expanded.real - C_explicit).norm() < 1e-4
