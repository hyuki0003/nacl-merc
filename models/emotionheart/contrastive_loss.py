
import torch
import torch.nn as nn
import torch.nn.functional as F

class MIM_loss(nn.Module):
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

    # def forward(self, c, embed, positive_valid_mask):
    #     B, N, _ = embed.shape
    #     device = embed.device
    #
    #     # L2 정규화
    #     c = F.normalize(c, p=2, dim=-1)
    #     embed = F.normalize(embed, p=2, dim=-1)
    #
    #     # 모든 노드와 모든 그래프 요약 간의 점수(코사인 유사도) 계산
    #     all_logits = torch.einsum('bnf,df->bnd', embed, c)
    #
    #     # Temperature 적용
    #     all_logits /= self.temperature
    #
    #     # CrossEntropyLoss를 위한 형태로 변환 (B*N, B)
    #     all_logits = all_logits.view(-1, B)
    #
    #     # 정답 레이블 생성
    #     labels = torch.arange(B, device=device).repeat_interleave(N)
    #
    #     # 손실 계산 (CrossEntropyLoss는 Softmax와 log를 모두 포함)
    #     loss = F.cross_entropy(all_logits, labels, reduction='none')
    #
    #     # 유효한 노드에 대해서만 마스킹
    #     valid_mask = positive_valid_mask.view(-1)
    #     masked_loss = loss[valid_mask]
    #
    #     return torch.mean(masked_loss) if masked_loss.numel() > 0 else torch.tensor(0.0).to(device)


    # def forward(self, c, embed, positive_valid_mask):
    #     B, N, _ = embed.shape
    #     device = embed.device
    #
    #     # ❗❗❗ L2 정규화(Normalization) 추가 ❗❗❗
    #     # 각 임베딩 벡터의 크기를 1로 만들어 줍니다.
    #     c = F.normalize(c, p=2, dim=-1)
    #     embed = F.normalize(embed, p=2, dim=-1)
    #
    #     # 1. 모든 노드와 모든 그래프 요약 간의 점수(코사인 유사도) 계산
    #     #    (이제 내적 연산이 코사인 유사도와 같아짐)
    #     all_logits = torch.einsum('bnf,df->bnd', embed, c)
    #
    #     # 2. Temperature 적용
    #     all_logits /= self.temperature
    #
    #     # 3. CrossEntropyLoss를 위한 형태로 변환
    #     all_logits = all_logits.view(-1, B)
    #
    #     # 4. 정답 레이블 생성
    #     labels = torch.arange(B, device=device).repeat_interleave(N)
    #
    #     # 5. 손실 계산
    #     loss = F.cross_entropy(all_logits, labels, reduction='none')
    #
    #     # 6. 유효한 노드에 대해서만 손실을 평균내기 위한 마스크 적용
    #     valid_mask = positive_valid_mask.view(-1)
    #     masked_loss = loss[valid_mask]
    #
    #     return torch.mean(masked_loss) if masked_loss.numel() > 0 else torch.tensor(0.0).to(device)

    def forward(self, c, embed, positive_valid_mask):

        B, N,_ = embed.shape
        device = embed.device

        # 1. 긍정 쌍 점수 계산 (기존과 동일)
        c_expanded = c.unsqueeze(1).expand_as(embed)
        positive_logits = torch.squeeze(self.f_k(embed, c_expanded), -1)  # Shape: (B, N)

        # 2. 부정 쌍 샘플링 (배치를 섞는 방식)
        # 배치 내에서 순서를 섞어 '잘못된' 노드 임베딩 생성
        shuffled_indices = torch.randperm(B).to(device)
        negative_embed = embed[shuffled_indices]

        # 3. 부정 쌍 점수 계산
        negative_logits = torch.squeeze(self.f_k(negative_embed, c_expanded), -1)  # Shape: (B, N)

        # 4. 긍정/부정 점수 및 레이블 통합
        # 긍정 점수와 부정 점수를 하나로 합침
        all_logits = torch.cat((positive_logits, negative_logits), dim=0)  # Shape: (2*B, N)
        all_logits /= self.temperature

        # 긍정 레이블(1)과 부정 레이블(0) 생성
        positive_labels = torch.ones_like(positive_logits)
        negative_labels = torch.zeros_like(negative_logits)
        all_labels = torch.cat((positive_labels, negative_labels), dim=0)  # Shape: (2*B, N)

        # 5. 손실 계산
        loss = self.BCEloss(all_logits, all_labels)

        # 6. 유효한 노드에 대해서만 손실을 평균내기 위한 마스크 적용
        valid_mask = positive_valid_mask.repeat(2, 1)  # 긍정/부정 쌍 모두에 마스크 적용

        # 마스크가 True인 위치의 loss 값만 가져와 평균 계산
        masked_loss = loss[valid_mask]

        return torch.mean(masked_loss) if masked_loss.numel() > 0 else torch.tensor(0.0).to(device)

    # def forward(self, c, embed, targets, positive_valid_mask):
    #     B, F = c.shape[0], c.shape[1]
    #     N = embed.shape[1]
    #
    #     # Apply bilinear function to graph summary and node features
    #     c_p = torch.unsqueeze(c, 1) # B*1*F
    #     c_p = c_p.expand_as(embed) # B*N*F
    #
    #     positive_sims = torch.squeeze(self.f_k(embed, c_p), -1) / self.temperature # B*N
    #
    #     # Expand graph_summary to match the shape for batch-wise comparison
    #     c_n = c.unsqueeze(1).unsqueeze(2).expand(B, B, N, F)# B*B*N*F
    #
    #     e_n = embed.unsqueeze(0).expand(B, B, N, F)# B*B*N*F
    #
    #     # Create mask to exclude positive samples (diagonal elements)
    #     negative_mask = ~torch.eye(B, dtype=bool).unsqueeze(-1).unsqueeze(-1).expand(B, B, N, F).to(embed.device)   # B*B*N*F
    #
    #     # Calculate negative samples using the bilinear function
    #     negative_sims = torch.squeeze(self.f_k(e_n[negative_mask].view(B,-1,F), c_n[negative_mask].view(B,-1,F)), -1) / self.temperature # B*B*N*F -> B*{B*(N-1)}*F ->  B*{(B-1)*N}
    #
    #     # Concatenate positive and negative samples
    #     scores = torch.cat((positive_sims, negative_sims), 1) # B*(B*N)
    #
    #     loss = self.BCEloss(scores, targets) # B*(B*N)
    #
    #     negative_valid_mask = positive_valid_mask.unsqueeze(0).expand(B,B,N) # B*N -> B*B*N
    #     negative_valid_mask = negative_valid_mask[negative_mask[:,:,:,0]].view(B,-1) # B*B*N -> B*{(B-1)*N}
    #     valid_mask = torch.cat((positive_valid_mask, negative_valid_mask), -1)
    #
    #     return torch.nanmean(loss[valid_mask])

class triadic_infoNCE_loss(nn.Module):
    def __init__(self, temperature):
        super(triadic_infoNCE_loss, self).__init__()
        self.temperature = temperature

    def batch_sim(self, z1: torch.Tensor, z2: torch.Tensor,eps=1e-12):

        # Normalize the vectors to avoid repeated computation of norms
        z1_norm = z1 / (torch.norm(z1, dim=-1, keepdim=True)+eps)  # Shape: (B, N, F)
        z2_norm = z2 / (torch.norm(z2, dim=-1, keepdim=True)+eps)  # Shape: (B, N, F)

        # Compute the cosine similarity using batch matrix multiplication
        cosine_similarity = torch.bmm(z1_norm, z2_norm.transpose(1, 2))  # Shape: (B, N, N)

        return cosine_similarity


    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3, mask, eps=1e-12):
        f = lambda x: torch.exp(x / self.temperature)

        within_sim_1 = self.batch_sim(z1, z1, eps)
        within_sim_1 = f(within_sim_1).clone()

        between_sim_1 = self.batch_sim(z1, z2, eps)
        between_sim_1 = f(between_sim_1).clone()

        between_sim_2 = self.batch_sim(z1, z3, eps)
        between_sim_2 = f(between_sim_2).clone()

        within_sim_1= within_sim_1.masked_fill(mask, 0.0)
        between_sim_1= between_sim_1.masked_fill(mask, 0.0)
        between_sim_2= between_sim_2.masked_fill(mask, 0.0)

        numerator = between_sim_1.diagonal(dim1=-2, dim2=-1) + between_sim_2.diagonal(dim1=-2, dim2=-1)
        denominator = (
            within_sim_1.sum(dim=-1) - within_sim_1.diagonal(dim1=-2, dim2=-1) # sum of negative samples within a reference graph without self-similarity
            + between_sim_1.sum(dim=-1) # sum of all samples including both negative and positive samples between the reference and the first comparison graph
            + between_sim_2.sum(dim=-1) # sum of all samples between the reference and the other comparison graph
        )
        loss = -torch.log((numerator+eps) / (denominator+eps))


        return torch.nanmean(loss)


    def forward(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, mask, eps=1e-12):
        l1 = self.loss(z1, z2, z3, mask, eps)
        l2 = self.loss(z2, z1, z3, mask, eps)
        l3 = self.loss(z3, z1, z2, mask, eps)

        loss = (l1+l2+l3)/3.

        return loss



class infoNCE_loss(nn.Module):
    def __init__(self, temperature):
        super(infoNCE_loss, self).__init__()
        self.temperature = temperature

    def batch_sim(self, z1: torch.Tensor, z2: torch.Tensor,eps=1e-12):

        # Normalize the vectors to avoid repeated computation of norms
        z1_norm = z1 / (torch.norm(z1, dim=-1, keepdim=True)+eps)  # Shape: (B, N, F)
        z2_norm = z2 / (torch.norm(z2, dim=-1, keepdim=True)+eps)  # Shape: (B, N, F)

        # Compute the cosine similarity using batch matrix multiplication
        cosine_similarity = torch.bmm(z1_norm, z2_norm.transpose(1, 2))  # Shape: (B, N, N)

        return cosine_similarity

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, mask, eps=1e-12):
        f = lambda x: torch.exp(x / self.temperature)

        # Clone하여 새로운 텐서 생성 (in-place 연산 방지)
        within_sim = f(self.batch_sim(z1, z1)).clone()
        between_sim = f(self.batch_sim(z1, z2)).clone()

        # in-place 연산 대신 `masked_fill_()` 사용
        within_sim = within_sim.masked_fill(mask, 0.0)
        between_sim = between_sim.masked_fill(mask, 0.0)

        numerator = between_sim.diagonal(dim1=-2, dim2=-1)
        denominator = between_sim.sum(dim=-1) + within_sim.sum(dim=-1) - within_sim.diagonal(dim1=-2, dim2=-1)

        loss = -torch.log((numerator + eps) / (denominator + eps))

        return torch.nanmean(loss)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, mask):
        ret = self.semi_loss(z1, z2, mask)
        return ret

# Version 1
# class NACL_loss(nn.Module):
#     def __init__(self, temperature):
#         super(NACL_loss, self).__init__()
#         self.temperature = temperature
#
#
#     def topK_masks(self, z, mask,n_classes, k=3):
#         """
#         Returns masks indicating the positions of top-K positive and negative samples,
#         with adjustments for cases where the number of valid samples is less than K.
#
#         Args:
#             sim (torch.Tensor): (B, N, N) similarity matrix.
#             mask (torch.Tensor): (B, N, N) padding mask (True = ignored values).
#             k (int): Number of top-K nearest neighbors.
#
#         Returns:
#             tuple(torch.Tensor, torch.Tensor):
#                 - positive_mask: (B, N, N) mask for positive pairs.
#                 - negative_mask: (B, N, N) mask for negative pairs.
#         """
#         _sim = torch.cdist(z,z, p=2)
#         B, N, _ = _sim.shape
#         sim = _sim.clone()  # sim dist and  Avoid modifying the original similarity matrix
#
#         self_mask = torch.eye(N, device=sim.device).bool().unsqueeze(0)  # (1, N, N), self-similarity mask
#         all_mask = mask | self_mask
#
#         sim.masked_fill_(all_mask, float('inf'))  # remove self similarity and padded one
#
#         # ✅ Count valid samples in each batch (excluding padding)
#         valid_sample_count = (~all_mask).sum(dim=-1).max(dim=-1).values # (B,) number of non-padding samples per batch
#
#         # ✅ Dynamic K: Adjust top-K for batches with fewer valid samples
#         dynamic_k = torch.where(valid_sample_count <= k, 1, k)
#
#         # ✅ Initialize positive_mask
#         positive_mask = torch.zeros((B, N, N), dtype=torch.bool, device=sim.device)  # (B, N, N)
#
#         for b in range(B):
#             if valid_sample_count[b] == 0:
#                 continue
#             top_k_indices = torch.topk(sim[b], dynamic_k[b].item(), dim=-1, largest=False).indices  # (N, dynamic_k)
#             positive_mask[b].scatter_(1, top_k_indices, True)  # Mark top-K positions as True
#
#         positive_mask.masked_fill_(all_mask, False)
#
#         # ✅ Create Negative Mask (excluding padding, self-similarity, and positive samples)
#         negative_mask = ~(all_mask | positive_mask)  # (B, N, N)
#         negative_mask.masked_fill_(self_mask, False)
#
#         return positive_mask, negative_mask
#
#
#     def batch_sim(self, z1: torch.Tensor, z2: torch.Tensor,eps=1e-12):
#
#         # Normalize the vectors to avoid repeated computation of norms
#         z1_norm = z1 / (torch.norm(z1, dim=-1, keepdim=True)+eps)  # Shape: (B, N, F)
#         z2_norm = z2 / (torch.norm(z2, dim=-1, keepdim=True)+eps)  # Shape: (B, N, F)
#
#         # Compute the cosine similarity using batch matrix multiplication
#         cosine_similarity = torch.bmm(z1_norm, z2_norm.transpose(1, 2))  # Shape: (B, N, N)
#
#         return cosine_similarity
#
#
#     def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, z1_all_mask, k, n_classes,eps=1e-12, mode=None):
#         f = lambda x: torch.exp(x / self.temperature)
#         within_sim = self.batch_sim(z2, z2)
#
#         within_sim_positive_mask, within_sim_negative_mask = self.topK_masks(z1, z1_all_mask, n_classes, k=k)
#
#         within_sim = f(within_sim)
#
#         within_positive_sim = within_sim*within_sim_positive_mask
#
#         within_negative_sim = within_sim*within_sim_negative_mask
#
#         numerator = within_positive_sim.sum(dim=-1)
#         denominator = numerator + within_negative_sim.sum(dim=-1)
#         #
#         # if torch.isinf(numerator).any():
#         #     print(f"🔥 Overflow detected in mode: {mode}")
#         #     print(f"Max value in numerator: {numerator.max()}")
#         #
#         # if torch.isnan(numerator).any():
#         #     print(f"🔥 Overflow detected in mode: {mode}")
#         #     print("🚨 NaN detected in numerator input tensors before loss calculation!")
#         # if torch.isnan(denominator).any():
#         #     print("🚨 NaN detected in denominator input tensors before loss calculation!")
#
#         loss = -torch.log((numerator+eps)/(denominator+eps))
#
#         return torch.nanmean(loss)
#
#
#     def forward(self, z1: torch.Tensor, z2: torch.Tensor, z1_all_mask, k, n_classes, mode=None):
#         return self.semi_loss(z1, z2, z1_all_mask, k, n_classes, mode=mode)

# Version 3

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
        Build positive mask of shape (B, N, N):
            row = anchor index i in z1
            col = candidate index n in z2

        Positive columns for row i:
            - n = i (direct joint positive), if z2[i] is valid
            - n in top-k neighbors of z1[i] within z1, mapped into z2 indices,
              while excluding invalid z2 candidates
        """
        B, N, _ = z1.shape
        device = z1.device

        # valid anchors are determined by z1
        valid_anchor = ~z1_mask  # (B, N)

        # source-modality KNN mining on z1
        metric_11 = self._pairwise_metric(z1, z1)  # (B, N, N)
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # (1, N, N)

        # invalid for KNN mining: source invalid rows/cols + self
        knn_invalid = z1_mask.unsqueeze(2) | z1_mask.unsqueeze(1) | eye
        metric_11 = metric_11.masked_fill(knn_invalid, -torch.finfo(metric_11.dtype).max)

        positive_mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)

        for b in range(B):
            valid_idx = torch.nonzero(valid_anchor[b], as_tuple=False).squeeze(-1)
            num_valid = valid_idx.numel()

            if num_valid == 0:
                continue

            # 1) direct cross-modal joint positive: z1[i] -> z2[i]
            joint_valid = valid_anchor[b] & (~z2_mask[b])  # both source anchor and target candidate valid
            positive_mask[b, joint_valid, torch.arange(N, device=device)[joint_valid]] = True

            # 2) neighborhood positives: neighbors found in z1, mapped by same indices into z2
            if num_valid > 1 and k > 0:
                k_b = min(k, num_valid - 1)

                submetric = metric_11[b, valid_idx, :]  # (num_valid, N)
                topk_idx = torch.topk(submetric, k=k_b, dim=-1, largest=True).indices  # (num_valid, k_b)

                row_idx = valid_idx[:, None].expand_as(topk_idx)

                # z2 invalid candidates must not become positives
                valid_neighbor_in_z2 = ~z2_mask[b, topk_idx]
                positive_mask[b, row_idx[valid_neighbor_in_z2], topk_idx[valid_neighbor_in_z2]] = True

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
        # logits: anchor = z1_i, candidate = z2_n
        logits = self._pairwise_metric(z1, z2) / self.temperature  # (B, N, N)

        # build positives from z1-neighborhood + direct joint pair
        positive_mask, valid_anchor = self._build_positive_mask(z1, z1_mask, z2_mask, k)

        # mask invalid target candidates only
        logits = logits.masked_fill(z2_mask.unsqueeze(1), -torch.finfo(logits.dtype).max)

        # log-softmax over candidate bank z2
        log_prob = F.log_softmax(logits, dim=-1)  # (B, N, N)

        # multi-positive soft target
        target = positive_mask.float()
        pos_count = target.sum(dim=-1, keepdim=True)  # (B, N, 1)

        # valid anchors must be valid in z1 and have at least one valid positive in z2
        valid_anchor = valid_anchor & (pos_count.squeeze(-1) > 0)

        target = target / pos_count.clamp_min(1.0)

        # cross-entropy with soft labels
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


# Version 2
# class NACL_loss(nn.Module):
#     """
#     Neighbor Alignment Contrastive Learning (NACL)
#
#     Anchor   : z2[:, i, :]   (target modality)
#     Candidates: z1[:, n, :]  (source modality)
#
#     Positive set for anchor i:
#         {i} U N_i^alpha
#     where N_i^alpha is the top-k self-excluded neighborhood of z1[:, i, :]
#     in the source modality space.
#
#     pair_mask: Bool tensor of shape (B, N, N)
#         True  -> invalid pair (padding / masked token / excluded candidate)
#         False -> valid pair
#     """
#
#     def __init__(
#         self,
#         temperature: float = 0.1,
#         similarity: str = "cosine",
#         eps: float = 1e-12,
#     ):
#         super().__init__()
#         assert similarity in {"cosine", "l2"}
#         self.temperature = temperature
#         self.similarity = similarity
#         self.eps = eps
#
#     def _valid_nodes(self, pair_mask: torch.Tensor) -> torch.Tensor:
#         """
#         pair_mask: (B, N, N), True = invalid
#         Assumes diagonal entries indicate whether each node itself is valid.
#         Returns:
#             valid_nodes: (B, N), True = valid node
#         """
#         return ~torch.diagonal(pair_mask, dim1=1, dim2=2)
#
#     def _pairwise_metric(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
#         """
#         Returns raw pairwise metric:
#             - cosine similarity if similarity == 'cosine'
#             - L2 distance      if similarity == 'l2'
#         Shape: (B, N, N)
#         """
#         if self.similarity == "cosine":
#             z1 = F.normalize(z1, p=2, dim=-1, eps=self.eps)
#             z2 = F.normalize(z2, p=2, dim=-1, eps=self.eps)
#             return torch.bmm(z1, z2.transpose(1, 2))
#         else:  # l2
#             return torch.cdist(z1, z2, p=2)
#
#     def _pairwise_logits(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
#         """
#         Returns similarity logits:
#             - cosine similarity
#             - negative L2 distance
#         Shape: (B, N, N)
#         """
#         metric = self._pairwise_metric(z1, z2)
#         if self.similarity == "cosine":
#             return metric
#         else:
#             return -metric  # smaller distance -> larger logit
#
#     def _build_positive_mask(
#         self,
#         ref: torch.Tensor,
#         pair_mask: torch.Tensor,
#         k: int,
#     ):
#         """
#         Build positive mask from source-modality neighborhoods.
#
#         ref      : (B, N, F), source modality embeddings used for neighbor mining
#         pair_mask: (B, N, N), True = invalid pair
#         k        : requested top-k
#
#         Returns:
#             positive_mask: (B, N, N), True at positive candidate positions
#             valid_nodes  : (B, N), True for valid node indices
#         """
#         B, N, _ = ref.shape
#         device = ref.device
#
#         valid_nodes = self._valid_nodes(pair_mask)  # (B, N)
#         eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # (1, N, N)
#
#         # Neighbor mining is performed within the source modality ref
#         metric = self._pairwise_metric(ref, ref)  # (B, N, N)
#
#         # Exclude invalid pairs and self-neighbors during KNN mining
#         if self.similarity == "cosine":
#             metric = metric.masked_fill(pair_mask | eye, -float("inf"))
#             largest = True
#         else:  # l2 distance
#             metric = metric.masked_fill(pair_mask | eye, float("inf"))
#             largest = False
#
#         positive_mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)
#
#         for b in range(B):
#             valid_idx = torch.nonzero(valid_nodes[b], as_tuple=False).squeeze(-1)
#             num_valid = valid_idx.numel()
#
#             if num_valid == 0:
#                 continue
#
#
#             positive_mask[b, valid_idx, valid_idx] = True
#
#             # top-k neighborhood positives (self excluded)
#             if num_valid > 1 and k > 0:
#                 k_b = min(k, num_valid - 1)
#                 submetric = metric[b, valid_idx, :]  # (num_valid, N)
#                 topk_idx = torch.topk(
#                     submetric, k=k_b, dim=-1, largest=largest
#                 ).indices  # (num_valid, k_b)
#
#                 row_idx = valid_idx[:, None].expand_as(topk_idx)
#                 positive_mask[b, row_idx, topk_idx] = True
#
#         # Remove any invalid positions
#         positive_mask &= ~pair_mask
#         return positive_mask, valid_nodes
#
#     def semi_loss(
#         self,
#         z1: torch.Tensor,
#         z2: torch.Tensor,
#         pair_mask: torch.Tensor,
#         k: int,
#     ) -> torch.Tensor:
#         """
#         z1: (B, N, F), source modality embeddings for neighborhood mining and candidate bank
#         z2: (B, N, F), target modality embeddings for anchors
#         pair_mask: (B, N, N), True = invalid pair (padding / masked token / excluded candidate)
#
#         Loss is computed for anchors in z2 against candidates in z1.
#         """
#         # logits(anchor=z2_i, candidate=z1_n)
#         logits = self._pairwise_logits(z2, z1) / self.temperature  # (B, N, N)
#
#         # positives = same-utterance joint pair + source-modality KNN neighbors
#
#         positive_mask, valid_nodes = self._build_positive_mask(z1, pair_mask, k)
#
#         # mask invalid candidates
#         logits = logits.masked_fill(pair_mask, -torch.finfo(logits.dtype).max)
#
#         # log-softmax over valid candidate bank
#         log_prob = F.log_softmax(logits, dim=-1)
#
#         # soft target over positives
#         target = positive_mask.float()
#         pos_count = target.sum(dim=-1, keepdim=True)  # (B, N, 1)
#
#         # valid anchors must be valid nodes and have at least one positive
#         valid_anchor = valid_nodes & (pos_count.squeeze(-1) > 0)
#
#         target = target / pos_count.clamp_min(1.0)
#
#         # cross-entropy with multi-positive soft targets
#         loss = -(target * log_prob).sum(dim=-1)  # (B, N)
#
#         if valid_anchor.any():
#             return loss[valid_anchor].mean()
#         else:
#             return logits.new_tensor(0.0)
#
#     def forward(
#         self,
#         z1: torch.Tensor,
#         z2: torch.Tensor,
#         pair_mask: torch.Tensor,
#         k: int,
#     ) -> torch.Tensor:
#         return self.semi_loss(z1, z2, pair_mask, k)



class neighbor_global_infoNCE_loss(nn.Module):
    def __init__(self, temperature):
        super(neighbor_global_infoNCE_loss, self).__init__()
        self.temperature = temperature

    def topK_masks(self, z, k=2):
        """
        Returns masks indicating the positions of top-K positive and negative samples,
        with adjustments for cases where the number of valid samples is less than K.

        Args:
            sim (torch.Tensor): (B, N, N) similarity matrix.
            mask (torch.Tensor): (B, N, N) padding mask (True = ignored values).
            k (int): Number of top-K nearest neighbors.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - positive_mask: (B, N, N) mask for positive pairs.
                - negative_mask: (B, N, N) mask for negative pairs.
        """
        sim = torch.cdist(z,z, p=2) # sim dist and  Avoid modifying the original similarity matrix
        B = sim.shape[0]
        self_mask = torch.eye(B, device=sim.device).bool()  # (1, N, N), self-similarity mask

        sim.masked_fill_(self_mask, float('inf'))
        # ✅ Initialize positive_mask
        positive_mask = torch.zeros_like(sim, dtype=torch.bool, device=sim.device)  # (B, B)

        for b in range(B):
            top_k_indices = torch.topk(sim[b], k, dim=-1, largest=False).indices  # (k)
            positive_mask[b].scatter_(-1, top_k_indices, True)  # Mark top-K positions as True

        # ✅ Create Negative Mask (excluding padding, self-similarity, and positive samples)
        negative_mask = ~positive_mask # (B, B)
        negative_mask = negative_mask.masked_fill(self_mask, False)

        return positive_mask, negative_mask

    def batch_sim(self, z: torch.Tensor, eps=1e-12):

        # Normalize the vectors to avoid repeated computation of norms
        z_norm = z / (torch.norm(z, dim=-1, keepdim=True) + eps)  # Shape: (B, D)

        # Compute the cosine similarity using batch matrix multiplication
        cosine_similarity = torch.mm(z_norm, z_norm.T)  # Shape: (B, N, N)

        return cosine_similarity


    def semi_loss(self, z: torch.Tensor,k,eps=1e-12):
        f = lambda x: torch.exp(x / self.temperature)
        within_sim = self.batch_sim(z)

        within_sim_positive_mask, within_sim_negative_mask = self.topK_masks(z, k=k)

        within_sim = f(within_sim).clone()

        within_positive_sim = within_sim*within_sim_positive_mask

        within_negative_sim = within_sim*within_sim_negative_mask

        numerator = within_positive_sim.sum(dim=-1)
        denominator = numerator + within_negative_sim.sum(dim=-1)

        loss = -torch.log((numerator+eps)/(denominator+eps))

        return torch.nanmean(loss)

    def forward(self, z:torch.Tensor, k):
        ret = self.semi_loss(z, k)

        return ret



class KNNContrastiveLoss_Utterance_Euclidian(nn.Module):
    def __init__(self, K=5, temperature=0.1):
        super(KNNContrastiveLoss_Utterance_Euclidian, self).__init__()
        self.K = K
        self.temperature = temperature

    def forward(self, anchor, modality_1, qmask, umask):
        """
        anchor: (B, U, D)  # 주어진 modality 표현
        modality_1: (B, U, D)  # 다른 modality 표현
        qmask: (B, U, num_speakers)  # 화자 정보 (one-hot encoding)
        umask: (B, U)  # 유효한 발화 (1이면 유효, 0이면 패딩)
        """
        device = anchor.device
        batch_size, num_utterances, hidden_dim = anchor.shape

        # 🔹 Step 1: Reshape (batch 차원 제거)
        anchor_flat = anchor.reshape(batch_size * num_utterances, hidden_dim)  # (B*U, D)
        modality_1_flat = modality_1.reshape(batch_size * num_utterances, hidden_dim)  # (B*U, D)

        # 🔹 Step 2: Compute Pairwise Euclidean Distances
        distances = torch.cdist(anchor_flat, anchor_flat, p=2)  # (B*U, B*U)

        distances.fill_diagonal_(1e10)  # 자기 자신은 제외

        # 🔹 Step 3: Masking (패딩된 발화 제거)
        umask_flat = umask.reshape(-1).to(device)  # (B*U,)
        valid_mask = umask_flat.unsqueeze(0) * umask_flat.unsqueeze(1).to(device)  # (B*U, B*U)
        distances = distances.masked_fill(valid_mask == 0, 1e10)  # 패딩된 발화는 고려하지 않음

        # 🔹 Step 4: Speaker Masking (동일 화자끼리만 비교)
        qmask_flat = qmask.reshape(batch_size * num_utterances, -1).to(device)  # (B*U, num_speakers)
        speaker_sim = torch.matmul(qmask_flat, qmask_flat.T)  # (B*U, B*U)
        distances = distances.masked_fill(speaker_sim == 0, 1e10)  # 다른 화자의 발화는 고려하지 않음

        # 🔹 Step 5: Find K-nearest neighbors (KNN)
        knn_indices = distances.argsort(dim=-1)[:, :self.K]  # 가장 가까운 K개의 이웃 선택

        # 🔹 Step 6: Compute Similarities
        modality_1_sim = torch.matmul(modality_1_flat, modality_1_flat.T) / self.temperature  # (B*U, B*U)

        # 🔹 Step 7: Create Positive Pair Mask (KNN 기반)
        pos_mask = torch.zeros_like(modality_1_sim, device=device)
        for i in range(pos_mask.shape[0]):
            pos_mask[i, knn_indices[i]] = 1  # KNN에서 선택된 이웃만 positive pair로 설정
        pos_mask = pos_mask * valid_mask  # 패딩된 발화는 제외

        # 🔹 Step 8: Compute InfoNCE Loss
        loss = self.info_nce_loss(modality_1_sim, pos_mask)

        return loss

    def info_nce_loss(self, sim_matrix, pos_mask):
        """
        InfoNCE Loss 계산 (Utterance 단위에서 Contrastive Loss 적용)
        """
        exp_sim = torch.exp(sim_matrix - sim_matrix.max(dim=-1, keepdim=True)[0])  # Numerical stability
        pos_exp_sim = exp_sim * pos_mask  # Positive Pair 유사도만 유지

        # 🔹 Ensure no division by zero
        pos_sum = pos_exp_sim.sum(dim=-1, keepdim=True) + 1e-8
        neg_sum = exp_sim.sum(dim=-1, keepdim=True) + 1e-8

        pos_loss = torch.log(pos_sum)  # log-sum-exp of positive pairs
        neg_loss = torch.log(neg_sum)  # log-sum-exp of all pairs

        return -torch.mean(pos_loss - neg_loss)  # Minimize loss


class KNNContrastiveLoss_Utterance_cossim(nn.Module):
    def __init__(self, K=5, temperature=0.1):
        super(KNNContrastiveLoss_Utterance_cossim, self).__init__()
        self.K = K
        self.temperature = temperature

    def forward(self, anchor, modality_1, qmask, umask):
        """
        anchor: (B, U, D)  # 주어진 modality 표현
        modality_1: (B, U, D)  # 다른 modality 표현
        qmask: (B, U, num_speakers)  # 화자 정보 (one-hot encoding)
        umask: (B, U)  # 유효한 발화 (1이면 유효, 0이면 패딩)
        """
        device = anchor.device
        batch_size, num_utterances, hidden_dim = anchor.shape

        # 🔹 Step 1: Reshape (batch 차원 제거)
        anchor_flat = anchor.reshape(batch_size * num_utterances, hidden_dim)  # (B*U, D)
        modality_1_flat = modality_1.reshape(batch_size * num_utterances, hidden_dim)  # (B*U, D)

        # 🔹 Step 2: Compute Cosine Similarity (Instead of Euclidean Distance)
        anchor_norm = F.normalize(anchor_flat, p=2, dim=1)  # 🔥 L2 정규화
        cosine_sim = torch.mm(anchor_norm, anchor_norm.T)  # 🔥 코사인 유사도 계산
        distances = 1 - cosine_sim  # 🔥 유사도를 거리 개념으로 변환

        distances.fill_diagonal_(1e10)  # 자기 자신 제외

        # 🔹 Step 3: Masking (패딩된 발화 제거)
        umask_flat = umask.reshape(-1).to(device)  # (B*U,)
        valid_mask = umask_flat.unsqueeze(0) * umask_flat.unsqueeze(1)  # (B*U, B*U)
        distances = distances.masked_fill(valid_mask == 0, 1e10)  # 패딩된 발화는 고려하지 않음

        # 🔹 Step 4: Speaker Masking (동일 화자끼리만 비교)
        qmask_flat = qmask.reshape(batch_size * num_utterances, -1).to(device)  # (B*U, num_speakers)
        speaker_sim = torch.matmul(qmask_flat, qmask_flat.T)  # (B*U, B*U)
        distances = distances.masked_fill(speaker_sim == 0, 1e10)  # 다른 화자의 발화는 고려하지 않음

        # 🔹 Step 5: Find K-nearest neighbors (KNN)
        knn_indices = distances.argsort(dim=-1)[:, :self.K]  # 가장 가까운 K개의 이웃 선택

        # 🔹 Step 6: Compute Similarities
        modality_1_sim = torch.matmul(modality_1_flat, modality_1_flat.T) / self.temperature  # (B*U, B*U)

        # 🔹 Step 7: Create Positive Pair Mask (KNN 기반)
        pos_mask = torch.zeros_like(modality_1_sim, device=device)
        for i in range(pos_mask.shape[0]):
            pos_mask[i, knn_indices[i]] = 1  # KNN에서 선택된 이웃만 positive pair로 설정
        pos_mask = pos_mask * valid_mask  # 패딩된 발화는 제외

        # 🔹 Step 8: Compute InfoNCE Loss
        loss = self.info_nce_loss(modality_1_sim, pos_mask)

        return loss

    def info_nce_loss(self, sim_matrix, pos_mask):
        """
        InfoNCE Loss 계산 (Utterance 단위에서 Contrastive Loss 적용)
        """
        exp_sim = torch.exp(sim_matrix - sim_matrix.max(dim=-1, keepdim=True)[0])  # Numerical stability
        pos_exp_sim = exp_sim * pos_mask  # Positive Pair 유사도만 유지

        # 🔹 Ensure no division by zero
        pos_sum = pos_exp_sim.sum(dim=-1, keepdim=True) + 1e-8
        neg_sum = exp_sim.sum(dim=-1, keepdim=True) + 1e-8

        pos_loss = torch.log(pos_sum)  # log-sum-exp of positive pairs
        neg_loss = torch.log(neg_sum)  # log-sum-exp of all pairs

        return -torch.mean(pos_loss - neg_loss)  # Minimize loss