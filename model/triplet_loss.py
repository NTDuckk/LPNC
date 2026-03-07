import torch
from torch import nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def tensor_euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def cosine_dist(x, y):
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection / (x_norm * y_norm)
    dist = (1.0 - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """
    For each anchor, find hardest positive and hardest negative.
    NOTE: assume each ID appears >= 2 times in a batch.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)

    N = dist_mat.size(0)

    # positive / negative mask
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # remove self-match from positives
    eye = torch.eye(N, dtype=torch.bool, device=dist_mat.device)
    is_pos = is_pos & (~eye)

    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True
    )
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True
    )

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        ind = (
            labels.new()
            .resize_as_(labels)
            .copy_(torch.arange(0, N, device=labels.device).long())
            .unsqueeze(0)
            .expand(N, N)
        )

        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data
        ).squeeze(1)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data
        ).squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Standard batch-hard triplet loss:
        L = max(d_p - d_n + m, 0)
    """

    def __init__(self, margin=0.3, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)

        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        # keep hook for compatibility, but PromptSG-style should use 0.0
        if self.hard_factor != 0.0:
            dist_ap = dist_ap * (1.0 + self.hard_factor)
            dist_an = dist_an * (1.0 - self.hard_factor)

        y = dist_an.new_ones(dist_an.size())
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss, dist_ap, dist_an