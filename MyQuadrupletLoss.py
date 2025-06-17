import torch
import torch.nn.functional as F

def compute_quadruplet_loss(x_a, x_p, x_n, x_t, alpha_1=0.2, alpha_2=0.5, lambda_weight=1.0):
    """
    Computes extended triplet loss with an additional trash negative sample.
    Args:
        x_a, x_p, x_n, x_t: torch tensors of shape [embedding_dim], L2-normalized
        alpha_1: margin for regular negative
        alpha_2: margin for trash negative
        lambda_weight: weighting for trash loss component
    Returns:
        total_loss: scalar float
        detailed_log: dict with individual components
    """
    # Ensure inputs are normalized
    x_a = F.normalize(x_a, dim=0)
    x_p = F.normalize(x_p, dim=0)
    x_n = F.normalize(x_n, dim=0)
    x_t = F.normalize(x_t, dim=0)

    # Cosine distances
    d_ap = 1 - F.cosine_similarity(x_a, x_p, dim=0)
    d_an = 1 - F.cosine_similarity(x_a, x_n, dim=0)
    d_at = 1 - F.cosine_similarity(x_a, x_t, dim=0)

    # Loss components
    loss_triplet = torch.clamp(d_ap - d_an + alpha_1, min=0.0)
    loss_trash = torch.clamp(d_ap - d_at + alpha_2, min=0.0)
    total_loss = loss_triplet + lambda_weight * loss_trash

    # Logging
    return total_loss.item(), {
        "d(a,p)": d_ap.item(),
        "d(a,n)": d_an.item(),
        "d(a,t)": d_at.item(),
        "Triplet Loss": loss_triplet.item(),
        "Trash Loss": loss_trash.item(),
        "Total Loss": total_loss.item()
    }
