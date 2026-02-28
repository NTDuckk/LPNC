import torch
from torch import nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def tensor_euclidean_dist(x, y):
    """
    compute euclidean distance between two matrix x and y
    with size (n1, d) and (n2, d) and type torch.tensor
    return a matrix (n1, n2)
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # legacy signature but still works
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)  # B, B
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection / (x_norm * y_norm)
    dist = (1.0 - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """
    For each anchor, find the hardest positive and negative sample.

    Robust version:
    - Works even when each label appears different times in the batch.
    - Avoids `view(N, -1)` on masked tensors (which crashes when #pos varies).
    """
    assert dist_mat.dim() == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    if labels.dim() == 1:
        labels_ = labels.view(N, 1)
    else:
        labels_ = labels
        if labels_.numel() != N:
            labels_ = labels.view(N, 1)

    # [N, N]
    is_pos = labels_.eq(labels_.t())
    is_neg = ~is_pos

    # Exclude self from positives
    diag = torch.eye(N, dtype=torch.bool, device=dist_mat.device)
    is_pos = is_pos & (~diag)

    # Hardest positive: max distance among positives
    dist_pos = dist_mat.clone()
    dist_pos[~is_pos] = -1e9
    dist_ap, p_inds = dist_pos.max(dim=1)  # [N], [N]

    # If an anchor has no positive (ID appears once), dist_ap will be -1e9
    no_pos = dist_ap.le(-1e8)
    if no_pos.any():
        dist_ap = dist_ap.clone()
        p_inds = p_inds.clone()
        dist_ap[no_pos] = 0.0
        # set to itself for safety
        p_inds[no_pos] = torch.arange(N, device=dist_mat.device, dtype=torch.long)[no_pos]

    # Hardest negative: min distance among negatives
    dist_neg = dist_mat.clone()
    dist_neg[~is_neg] = 1e9
    dist_an, n_inds = dist_neg.min(dim=1)  # [N], [N]

    no_neg = dist_an.ge(1e8)
    if no_neg.any():
        dist_an = dist_an.clone()
        n_inds = n_inds.clone()
        dist_an[no_neg] = 0.0
        n_inds[no_neg] = torch.arange(N, device=dist_mat.device, dtype=torch.long)[no_neg]

    if return_inds:
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.3):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)  # B,B
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


class RankingLoss:

    def __init__(self):
        pass

    def _label2similarity(sekf, label1, label2):
        '''
        compute similarity matrix of label1 and label2
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [n]
        :return: torch.Tensor, [m, n], {0, 1}
        '''
        m, n = len(label1), len(label2)
        l1 = label1.view(m, 1).expand([m, n])
        l2 = label2.view(n, 1).expand([n, m]).t()
        similarity = l1 == l2
        return similarity

    def _batch_hard(self, mat_distance, mat_similarity, more_similar):

        if more_similar == 'smaller':
            sorted_mat_distance, _ = torch.sort(
                mat_distance + (-9999999.0) * (1 - mat_similarity),
                dim=1, descending=True
            )
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, _ = torch.sort(
                mat_distance + (9999999.0) * (mat_similarity),
                dim=1, descending=False
            )
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n

        elif more_similar == 'larger':
            sorted_mat_distance, _ = torch.sort(
                mat_distance + (9999999.0) * (1 - mat_similarity),
                dim=1, descending=False
            )
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, _ = torch.sort(
                mat_distance + (-9999999.0) * (mat_similarity),
                dim=1, descending=True
            )
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n


class PlasticityLoss(RankingLoss):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    '''

    def __init__(self, margin, metric, if_l2='euclidean'):
        '''
        :param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
        :param bh: batch hard
        :param metric: l2 distance or cosine distance
        '''
        self.margin = margin
        self.margin_loss = nn.MarginRankingLoss(margin=margin)
        self.metric = metric
        self.if_l2 = if_l2

    def __call__(self, emb1, emb2, emb3, label1, label2, label3):
        '''
        :param emb1: torch.Tensor, [m, dim]
        :param emb2: torch.Tensor, [n, dim]
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [b]
        :return:
        '''

        if self.metric == 'cosine':
            mat_dist = cosine_dist(emb1, emb2)
            mat_dist = torch.log(1 + torch.exp(mat_dist))
            mat_sim = self._label2similarity(label1, label2)
            hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            mat_dist = cosine_dist(emb1, emb3)
            mat_dist = torch.log(1 + torch.exp(mat_dist))
            mat_sim = self._label2similarity(label1, label3)
            _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            margin_label = -torch.ones_like(hard_p)

        elif self.metric == 'euclidean':
            if self.if_l2:
                emb1 = F.normalize(emb1)
                emb2 = F.normalize(emb2)
            mat_dist = tensor_euclidean_dist(emb1, emb2)
            mat_dist = torch.log(1 + torch.exp(mat_dist))
            mat_sim = self._label2similarity(label1, label2)
            hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

            mat_dist = tensor_euclidean_dist(emb1, emb3)
            mat_dist = torch.log(1 + torch.exp(mat_dist))
            mat_sim = self._label2similarity(label1, label3)
            _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

            margin_label = torch.ones_like(hard_p)

        return self.margin_loss(hard_n, hard_p, margin_label)