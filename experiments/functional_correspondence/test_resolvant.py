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


def solve_C_explicit_laplacian(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3):
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


def test_expanded(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3, resolvant_gamma=0.5):
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

    # instead of the mask, we need to compute the resolvant operator
    scaling_factor = max(torch.max(evals_x), torch.max(evals_y))
    evals_gamma1 = (evals_x / scaling_factor) ** resolvant_gamma
    evals_gamma2 = (evals_y / scaling_factor) ** resolvant_gamma

    rn1_re = evals_gamma1 / (evals_gamma1.square() + 1)
    rn1_im = 1 / (evals_gamma1.square() + 1)

    rn2_re = evals_gamma2 / (evals_gamma2.square() + 1)
    rn2_im = 1 / (evals_gamma2.square() + 1)

    # inverse of a diagonal matrix is just the inverse of each entry on the diagonal
    lx_Ik_re = torch.kron(torch.diag(rn1_re), torch.eye(k, device=A.device))
    Ik_ly_re = torch.kron(torch.eye(k, device=A.device), torch.diag(rn2_re))
    lx_Ik_im = torch.kron(torch.diag(rn1_im), torch.eye(k, device=A.device))
    Ik_ly_im = torch.kron(torch.eye(k, device=A.device), torch.diag(rn2_im))
    # Delta = (lx_Ik - Ik_ly)
    Delta_re = (Ik_ly_re - lx_Ik_re)
    Delta_im = (Ik_ly_im - lx_Ik_im)

    first = At_Ik.T @ At_Ik
    second = Delta_re.T @ Delta_re + Delta_im.T @ Delta_im
    op = first + lambda_param * second
    rhs = At_Ik.T @ vec_B

    # we only care about the real part of the system
    C = torch.linalg.solve(op, rhs)

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


def debug_resolvant(evals_x, evals_y, C_lap, resolvant_gamma=0.5):
    D = get_mask(evals_x.unsqueeze(0), evals_y.unsqueeze(0), resolvant_gamma)
    evals_x = evals_x.squeeze(0)
    evals_y = evals_y.squeeze(0)

    scaling_factor = max(torch.max(evals_x), torch.max(evals_y))

    evals_gamma1 = (evals_x / scaling_factor) ** resolvant_gamma
    evals_gamma2 = (evals_y / scaling_factor) ** resolvant_gamma

    rn1_re = evals_gamma1 / (evals_gamma1.square() + 1)
    rn1_im = 1 / (evals_gamma1.square() + 1)

    rn2_re = evals_gamma2 / (evals_gamma2.square() + 1)
    rn2_im = 1 / (evals_gamma2.square() + 1)

    t_real = C_lap @ torch.diag(rn1_re) - torch.diag(rn2_re) @ C_lap
    t_imag = C_lap @ torch.diag(rn1_im) - torch.diag(rn2_im) @ C_lap
    full_term = torch.norm(t_real) + torch.norm(t_imag)

    res = torch.norm(torch.mul(D, C_lap ** 2, 'fro'))
    # should be small, but currently is not.
    # there is still a bug
    print("Resolvant diff: ", torch.norm(full_term - res))


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
    C_lap = solve_C_explicit_laplacian(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3)
    # debug_resolvant(evals_x, evals_y, C_lap)
    for i in np.linspace(0, 100, 20):
        C_expanded = test_expanded(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=i)
        C_explicit = test_explicit(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=i).squeeze(0)
        print((C_expanded.real - C_explicit).norm())
