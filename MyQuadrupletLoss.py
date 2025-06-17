import torch
import torch.nn.functional as F

class MyQuadrupletLoss:
    def __init__(self, alpha_1=0.2, alpha_2=0.5, lambda_weight=1.0):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_weight = lambda_weight

    def compute_single(self, x_a, x_p, x_n, x_t):
        """
        Compute loss for a single quadruplet (non-batched).
        Args:
            x_a, x_p, x_n, x_t: torch tensors of shape [D]
        Returns:
            total_loss (float), log (dict)
        """
        x_a = F.normalize(x_a, dim=0)
        x_p = F.normalize(x_p, dim=0)
        x_n = F.normalize(x_n, dim=0)
        x_t = F.normalize(x_t, dim=0)

        d_ap = 1 - F.cosine_similarity(x_a, x_p, dim=0)
        d_an = 1 - F.cosine_similarity(x_a, x_n, dim=0)
        d_at = 1 - F.cosine_similarity(x_a, x_t, dim=0)

        loss_triplet = torch.clamp(d_ap - d_an + self.alpha_1, min=0.0)
        loss_trash = torch.clamp(d_ap - d_at + self.alpha_2, min=0.0)
        total_loss = loss_triplet + self.lambda_weight * loss_trash

        return total_loss.item(), {
            "d(a,p)": d_ap.item(),
            "d(a,n)": d_an.item(),
            "d(a,t)": d_at.item(),
            "Triplet Loss": loss_triplet.item(),
            "Trash Loss": loss_trash.item(),
            "Total Loss": total_loss.item()
        }

    def compute_batch(self, x_a, x_p, x_n, x_t):
        """
        Compute loss for a batch of quadruplets.
        Args:
            x_a, x_p, x_n, x_t: torch tensors of shape [B, D]
        Returns:
            total_loss (scalar tensor)
        """
        x_a = F.normalize(x_a, dim=1)
        x_p = F.normalize(x_p, dim=1)
        x_n = F.normalize(x_n, dim=1)
        x_t = F.normalize(x_t, dim=1)

        d_ap = 1 - F.cosine_similarity(x_a, x_p, dim=1)
        d_an = 1 - F.cosine_similarity(x_a, x_n, dim=1)
        d_at = 1 - F.cosine_similarity(x_a, x_t, dim=1)

        loss_triplet = torch.clamp(d_ap - d_an + self.alpha_1, min=0.0)
        loss_trash = torch.clamp(d_ap - d_at + self.alpha_2, min=0.0)
        total_loss = loss_triplet + self.lambda_weight * loss_trash

        return total_loss.mean()
