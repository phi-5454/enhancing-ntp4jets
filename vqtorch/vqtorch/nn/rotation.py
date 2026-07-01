import torch
import torch.nn.functional as F

def rotation_trick(z_e, z_q, eps=1e-6):

    norm_e = z_e.norm(dim=-1, keepdim=True).clamp(min=eps)
    norm_q = z_q.norm(dim=-1, keepdim=True).clamp(min=eps)

    u = z_e / norm_e
    q = z_q / norm_q

    w = F.normalize(
        (u + q).detach(),
        dim=-1
    )

    e = z_e.unsqueeze(-2)

    w_col = w.unsqueeze(-1)
    w_row = w.unsqueeze(-2)

    u_col = u.detach().unsqueeze(-1)
    q_row = q.detach().unsqueeze(-2)

    rotated = (
        e
        - 2.0 * (e @ w_col @ w_row)
        + 2.0 * (e @ u_col @ q_row)
    )

    rotated = rotated.squeeze(-2)

    rotated = rotated * (
        norm_q / norm_e
    ).detach()

    return rotated