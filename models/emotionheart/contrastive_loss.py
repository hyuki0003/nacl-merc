import torch
import torch.nn as nn
import torch.nn.functional as F


class MIM_loss(nn.Module):
    """DGI-style mutual-information maximization between a graph summary and node embeddings."""

    def __init__(self, n_h, temperature=1.0):
        super(MIM_loss, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.temperature = temperature
        self.BCEloss = nn.BCEWithLogitsLoss(reduction='none')

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, embed, positive_valid_mask):
        B, N, _ = embed.shape
        device = embed.device

        # positive pairs: summary vs its own nodes
        c_expanded = c.unsqueeze(1).expand_as(embed)
        positive_logits = torch.squeeze(self.f_k(embed, c_expanded), -1)  # (B, N)

        # negative pairs: summary vs nodes of a shuffled batch element
        shuffled_indices = torch.randperm(B).to(device)
        negative_embed = embed[shuffled_indices]
        negative_logits = torch.squeeze(self.f_k(negative_embed, c_expanded), -1)  # (B, N)

        all_logits = torch.cat((positive_logits, negative_logits), dim=0)  # (2B, N)
        all_logits /= self.temperature

        positive_labels = torch.ones_like(positive_logits)
        negative_labels = torch.zeros_like(negative_logits)
        all_labels = torch.cat((positive_labels, negative_labels), dim=0)  # (2B, N)

        loss = self.BCEloss(all_logits, all_labels)

        valid_mask = positive_valid_mask.repeat(2, 1)
        masked_loss = loss[valid_mask]

        return torch.mean(masked_loss) if masked_loss.numel() > 0 else torch.tensor(0.0).to(device)


class infoNCE_loss(nn.Module):
    """Standard cross-modal InfoNCE (CLIP/VATT-style) with same-index positives."""

    def __init__(self, temperature):
        super(infoNCE_loss, self).__init__()
        self.temperature = temperature

    def batch_sim(self, z1: torch.Tensor, z2: torch.Tensor, eps=1e-12):
        z1_norm = z1 / (torch.norm(z1, dim=-1, keepdim=True) + eps)  # (B, N, F)
        z2_norm = z2 / (torch.norm(z2, dim=-1, keepdim=True) + eps)  # (B, N, F)
        return torch.bmm(z1_norm, z2_norm.transpose(1, 2))  # (B, N, N)

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, mask, eps=1e-12):
        f = lambda x: torch.exp(x / self.temperature)

        within_sim = f(self.batch_sim(z1, z1)).clone()
        between_sim = f(self.batch_sim(z1, z2)).clone()

        within_sim = within_sim.masked_fill(mask, 0.0)
        between_sim = between_sim.masked_fill(mask, 0.0)

        numerator = between_sim.diagonal(dim1=-2, dim2=-1)
        denominator = between_sim.sum(dim=-1) + within_sim.sum(dim=-1) - within_sim.diagonal(dim1=-2, dim2=-1)

        loss = -torch.log((numerator + eps) / (denominator + eps))

        return torch.nanmean(loss)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, mask):
        return self.semi_loss(z1, z2, mask)


class NACL_loss(nn.Module):
    """
    Neighbor Alignment Contrastive Learning (NACL)

    Anchor      : z1[:, i, :]   (source modality)
    Candidates  : z2[:, n, :]   (target modality)

    Positive set for anchor i:
        {i} U N_i^(z1)
    where N_i^(z1) is the top-k self-excluded neighborhood of z1[:, i, :]
    in the source-modality space.

    Masks:
        z1_mask: (B, N), True = invalid source node (padding / masked)
        z2_mask: (B, N), True = invalid target node (padding / masked)
    """

    def __init__(self, temperature: float = 0.1, similarity: str = "cosine", eps: float = 1e-12):
        super().__init__()
        assert similarity in {"cosine", "l2"}
        self.temperature = temperature
        self.similarity = similarity
        self.eps = eps

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z, p=2, dim=-1, eps=self.eps)

    def _pairwise_metric(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Returns pairwise similarity score matrix of shape (B, N, N).

        cosine: larger is better
        l2    : returns negative distance so that larger is better
        """
        if self.similarity == "cosine":
            z1 = self._normalize(z1)
            z2 = self._normalize(z2)
            return torch.bmm(z1, z2.transpose(1, 2))
        else:
            return -torch.cdist(z1, z2, p=2)

    def _build_positive_mask(
        self,
        z1: torch.Tensor,
        z1_mask: torch.Tensor,
        z2_mask: torch.Tensor,
        k: int,
    ):
        """
        Build positive mask of shape (B, N, N).

        row = anchor index i in z1
        col = candidate index n in z2

        Positive columns for row i:
            - n = i (direct joint positive), if z1[i] and z2[i] are both valid
            - n in top-k neighbors of z1[i] within VALID z1 nodes only,
              mapped by same indices into z2, while excluding invalid z2 candidates
        """
        B, N, _ = z1.shape
        device = z1.device

        valid_anchor = ~z1_mask                     # (B, N)
        target_valid = ~z2_mask                     # (B, N)

        positive_mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)

        # ------------------------------------------------------------------
        # 1) direct cross-modal joint positive: z1[i] -> z2[i]
        # ------------------------------------------------------------------
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)   # (1, N, N)
        joint_valid = valid_anchor & target_valid                          # (B, N)
        positive_mask |= eye & joint_valid.unsqueeze(-1)

        # ------------------------------------------------------------------
        # 2) neighborhood positives from z1-space
        #    per batch b, only k_b = min(k, num_valid_b - 1) neighbors are used
        # ------------------------------------------------------------------
        if k > 0 and N > 1:
            metric_11 = self._pairwise_metric(z1, z1)  # (B, N, N)

            # invalid for KNN mining: invalid source row/col + self
            knn_invalid = z1_mask.unsqueeze(2) | z1_mask.unsqueeze(1) | eye
            metric_11 = metric_11.masked_fill(knn_invalid, -torch.finfo(metric_11.dtype).max)

            # global max top-k size just for one-shot vectorized extraction
            k_global = min(k, N - 1)
            if k_global > 0:
                topk_indices = torch.topk(metric_11, k=k_global, dim=-1, largest=True).indices  # (B, N, k_global)

                # number of valid source nodes per batch
                num_valid = valid_anchor.sum(dim=1)  # (B,)

                # k_b = min(k, num_valid_b - 1); if num_valid_b <= 1, use 0 neighbors
                k_per_batch = torch.clamp(num_valid - 1, min=0)
                k_per_batch = torch.minimum(
                    k_per_batch,
                    torch.full_like(k_per_batch, k_global)
                )  # (B,)

                # select only first k_b neighbors for each batch b
                take_mask = (
                    torch.arange(k_global, device=device)
                    .view(1, 1, k_global)
                    < k_per_batch.view(B, 1, 1)
                )  # (B, 1, k_global) broadcast over anchor rows

                # keep only first k_b entries per batch/row
                selected_neighbor_mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)
                selected_neighbor_mask.scatter_(2, topk_indices, take_mask.expand(B, N, k_global))

                # prune:
                # - anchor row must be valid in z1
                # - candidate col must be valid in z2
                selected_neighbor_mask &= valid_anchor.unsqueeze(2)
                selected_neighbor_mask &= target_valid.unsqueeze(1)

                positive_mask |= selected_neighbor_mask

        return positive_mask, valid_anchor

    def semi_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z1_mask: torch.Tensor,
        z2_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        z1:      (B, N, F), source modality embeddings (anchors + neighbor mining source)
        z2:      (B, N, F), target modality embeddings (candidate bank)
        z1_mask: (B, N), True = invalid source anchor / source node
        z2_mask: (B, N), True = invalid target candidate
        """
        logits = self._pairwise_metric(z1, z2) / self.temperature  # (B, N, N)

        positive_mask, valid_anchor = self._build_positive_mask(z1, z1_mask, z2_mask, k)

        # invalid target candidates are excluded from normalization pool
        logits = logits.masked_fill(z2_mask.unsqueeze(1), -torch.finfo(logits.dtype).max)

        log_prob = F.log_softmax(logits, dim=-1)  # (B, N, N)

        target = positive_mask.float()
        pos_count = target.sum(dim=-1, keepdim=True)  # (B, N, 1)

        # valid anchor must be valid in z1 and have at least one positive in z2
        valid_anchor = valid_anchor & (pos_count.squeeze(-1) > 0)

        target = target / pos_count.clamp_min(1.0)

        loss = -(target * log_prob).sum(dim=-1)  # (B, N)

        if valid_anchor.any():
            return loss[valid_anchor].mean()
        else:
            return logits.new_tensor(0.0)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z1_mask: torch.Tensor,
        z2_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        return self.semi_loss(z1, z2, z1_mask, z2_mask, k)
