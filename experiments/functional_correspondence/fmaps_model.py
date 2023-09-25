import os
import sys

import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))
import diffusion_net


def compute_correspondence_iterative(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3):
    """
    Computes the functional map correspondence matrix C given features from two shapes.

    Has no trainable parameters.
    """

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
    lmbda = cp.Parameter(1, nonneg=True)
    objective = cp.Minimize(cp.norm(C @ A_var - B_var, 'fro') +
                            lmbda * cp.norm(C @ cp.diag(L1_var) - cp.diag(L2_var) @ C, 'fro'))

    problem = cp.Problem(objective)

    cvxpylayer = CvxpyLayer(problem, parameters=[A_var, B_var, L1_var, L2_var, lambda_param],
                            variables=[C])
    C, = cvxpylayer(A, B, evals_x.squeeze(0), evals_y.squeeze(0))

    return C.to(feat_x.device).unsqueeze(0)


def compute_correspondence_explicit(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3):
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

        self.n_fmap = 30
        self.input_features = input_features
        self.lambda_param = lambda_param

    def forward(self, shape1, shape2):
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
        C_iterative= compute_correspondence_iterative(feat1, feat2, evals1, evals2, evecs_trans1, evecs_trans2, lambda_param=self.lambda_param)
        print("iterative", time.time()-t0)
        t0 = time.time()
        C_explicit= compute_correspondence_explicit(feat1, feat2, evals1, evals2, evecs_trans1, evecs_trans2, lambda_param=self.lambda_param)
        print("explicit", time.time() - t0)

        diff = torch.mean(torch.abs(C_iterative - C_explicit))
        print("diff: ", diff)

        C_pred = C_iterative

        return C_pred, feat1, feat2
