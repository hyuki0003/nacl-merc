# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from .contrastive_loss import NACL_loss, MIM_loss, infoNCE_loss

logger = logging.getLogger(__name__)
logging.basicConfig(force=True, level=logging.INFO)

class FinetuneModel(BaseFairseqModel):
    def __init__(self,
                 args,
                 model,
                 encoder_embed_dim=None,
                 modalities=None,
                 intra_dataset=True
                 ):
        super().__init__()
        assert args.encoder_embed_dim == encoder_embed_dim, f"pretrained encoder dim {encoder_embed_dim}, but got {args.encoder_embed_dim}"
        # assert args.modalities == modalities, f"pretrained encoder use modality {modalities}, but got {args.modalities}"

        self.args = args
        self.pretrained_encoder = None
        self.encoder_embed_dim = encoder_embed_dim
        self.modalities = modalities
        self.n_modalities = len(self.modalities)
        self.is_intra_dataset = intra_dataset

        self.model = model
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(args.encoder_embed_dim)


        for name, param in self.model.named_parameters():
            print(f"Layer '{name}' - Parameter '{param.requires_grad}' ")


        if intra_dataset == False:
            data_embedding_dims = args.dataset_embedding_dims[args.dataset]
            if 'a' in args.modalities:
                self.input_projection_a = nn.Sequential(
                    nn.Linear(data_embedding_dims['a'], args.encoder_embed_dim),
                    # self.activation,
                    # nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
                )
    
            if 't' in args.modalities:
                self.input_projection_t = nn.Sequential(
                    nn.Linear(data_embedding_dims['t'], args.encoder_embed_dim),
                    # self.activation,
                    # nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
                )

            if 'v' in args.modalities:
                self.input_projection_v = nn.Sequential(
                    nn.Linear(data_embedding_dims['v'], args.encoder_embed_dim),
                    # self.activation,
                    # nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
                )
            for name, module in list(self.model.named_children()):
                if not name.startswith("encoder"):
                    delattr(self.model, name)

            # self.freeze_encoder("encoder.graph_encoder.layers.1")
            # self.freeze_encoder("encoder.graph_encoder.layers.0")


        else:
            for name, module in list(self.model.named_children()):
                if not name.startswith("encoder") and not name.startswith("input_projection"):
                    delattr(self.model, name)


        for name, param in self.model.named_parameters():
            print(f"Layer '{name}' - Parameter '{param.requires_grad}' ")


        if args.do_NACL == True:
            self.NACLloss = NACL_loss(args.temperature)
        if args.do_DGI == True:
            # self.DGI_projection = nn.Sequential(
            #     nn.LayerNorm(args.encoder_embed_dim),
            #     nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim*self.n_modalities),
            #     self.activation,
            #     nn.Linear(args.encoder_embed_dim*self.n_modalities, args.encoder_embed_dim)
            # )
            self.DGIloss = MIM_loss(args.encoder_embed_dim, args.temperature)
        if args.do_VATT == True:
            # self.CLIP_projection = nn.Sequential(
            #     nn.LayerNorm(args.encoder_embed_dim),
            #     nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim*self.n_modalities),
            #     self.activation,
            #     nn.Linear(args.encoder_embed_dim*self.n_modalities, args.encoder_embed_dim)
            # )
            self.VATTloss = infoNCE_loss(args.temperature)
            

        # --- [NEW] Token-level Modality Attention Fusion ---
        # Ref: Standard Attention Bottleneck / Soft-Attention mechanism for late fusion.
        # Dynamically calculates attention weights per token for each modality.
        # self.attention_fusion = nn.Sequential(
        #     nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim // 4),
        #     self.activation,
        #     nn.Linear(args.encoder_embed_dim // 4, 1)
        # )

        # --- [OLD] Static Linear Fusion ---
        # self.linear_fusion = nn.Linear(self.n_modalities, 1)
        self.linear_fusion = nn.Sequential(
            nn.LayerNorm(self.n_modalities * args.encoder_embed_dim),
            nn.Linear(self.n_modalities * args.encoder_embed_dim, args.encoder_embed_dim),
        )

        self.classifier = None
        if self.n_modalities == 1:
            self.classifier = nn.Sequential(
                self.activation,
                nn.Dropout(args.dropout),
                nn.Linear(args.encoder_embed_dim, args.num_classes)
            )
        else:
            # self.classifier = nn.Sequential(
            #     nn.Dropout(args.dropout),
            #     nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim // 4),
            #     self.activation,
            #     # nn.Dropout(args.dropout),
            #     nn.Linear(args.encoder_embed_dim // 4, args.num_classes)
            # )
            self.classifier = nn.Sequential(
                self.activation,
                nn.Dropout(args.dropout),
                # nn.Linear(args.encoder_embed_dim, args.sub_classifier_dim),
                # self.activation,
                # nn.Linear(args.sub_classifier_dim, args.num_classes)
                nn.Linear(args.encoder_embed_dim, args.num_classes)
            )

        # --- [Unimodal Auxiliary] Modality-specific classifiers ---
        # Ref: "What Makes Training Multi-modal Classification Networks Hard?" (CVPR 2022)
        #      "MISA: Modality-Invariant and -Specific Representations" (ACM MM 2020)
        # Purpose: Force each modality branch to independently learn discriminative features
        #          (prevents Modality Laziness / Imbalance in joint training).
        # if self.n_modalities >1:
        #     self.unimodal_classifiers = nn.ModuleList([
        #         nn.Sequential(
        #             self.activation,
        #             nn.Dropout(args.dropout),
        #             nn.Linear(args.encoder_embed_dim, args.num_classes)
        #         ) for _ in range(self.n_modalities)
        #     ])


    def freeze_encoder(self,param_name):
        for name, param in self.model.named_parameters():
            # param.requires_grad = False
            # print(f" Layer '{name}' is frozen.")
            if name.startswith(param_name):
                param.requires_grad = False
                print(f" Layer '{name}' is frozen.")
            # if not name.startswith("encoder.graph_encoder.layers.3"):
            #     param.requires_grad = False
            #     print(f" Layer '{name}' is frozen.")


    def forward(self, data, train=False):
        if self.args.dataset == "iemocap":
            data['utterance_order'] = torch.zeros_like(data['utterance_order'], dtype=torch.long,
                                                       device=data['utterance_order'].device)

        within_modality_loss = 0.
        between_modality_loss = 0.
        cross_entropy_loss = 0.
        unimodal_ce_loss = 0.

        mask = data['mask'].clone()
        B, N = mask.shape
        inverted_mask = mask.logical_not()

        sim_mask = mask.unsqueeze(1) | mask.unsqueeze(2)
        data['mask'] = mask.repeat(1, self.n_modalities)

        proj_list = []
        if self.is_intra_dataset == False:
            if 'a' in self.modalities:
                proj_list.append(self.input_projection_a(data['audio']))
            if 't' in self.modalities:
                proj_list.append(self.input_projection_t(data['text']))
            if 'v' in self.modalities:
                proj_list.append(self.input_projection_v(data['visual']))
        else:
            if 'a' in self.modalities:
                proj_list.append(self.model.input_projection_a(data['audio']))
            if 't' in self.modalities:
                proj_list.append(self.model.input_projection_t(data['text']))
            if 'v' in self.modalities:
                proj_list.append(self.model.input_projection_v(data['visual']))

        org_x = torch.stack(proj_list, dim=1).permute(0,2,3,1).contiguous()
        proj = torch.stack(proj_list, dim=1).view(B, -1, self.args.encoder_embed_dim)
        data['x'] = self.layer_norm(proj)

        representation = self.model.encoder(data, self.modalities)

        if self.args.specific and not self.args.hybrid:
            cls = representation[:, :, 0, :]
            nodes = representation[:, :, 1:, :]  # B, M, N, D
            graphs = nodes.permute(0, 2, 3, 1).contiguous()[inverted_mask, :] # B, M, N, D -> B, N, D, M -> B*valid N, D, M
            fused_emb = nodes.permute(0, 2, 1, 3).contiguous().view(B, N, -1) # B, N, D*M

        else:
            cls = representation[:, 0, :]
            nodes = representation[:, 1:, :]  # real nodes (i.e., utterance tokens)

            fused_emb = nodes.reshape(B, self.n_modalities, N, -1)
            graphs = fused_emb.permute(0, 2, 3, 1).contiguous()[inverted_mask, :]
            fused_emb = fused_emb.permute(0, 2, 1, 3).contiguous().view(B, N, -1)

        if train:
            if self.args.do_VATT:
                cnt = 0 if self.n_modalities != 1 else 1
                modals = nodes
                if not self.args.specific or self.args.hybrid:
                    modals = nodes.reshape(B, self.n_modalities, N, -1)
                for i in range(self.n_modalities):
                    for j in range(self.n_modalities):
                        if i == j:
                            continue
                        cnt += 1
                        source = modals[:, i, :, :]
                        target = modals[:, j, :, :]
                        # source +== self.CLIP_projection(source)
                        # target +== self.CLIP_projection(target)

                        between_modality_loss += self.VATTloss(source, target, sim_mask)
                between_modality_loss /= cnt
                between_modality_loss *= self.args.VATT_lambda

            if self.args.do_DGI:
                if self.args.specific and not self.args.hybrid:
                    # targets = torch.cat([torch.ones((B,N)), torch.zeros((B,(B-1)*N))], dim=-1).to(nodes.device)
                    for i in range(self.n_modalities):
                        m_cls = cls[:, i, :]
                        m_embed = nodes[:, i, :, :]
                        # m_cls += self.DGI_projection(m_cls)
                        # m_embed += self.DGI_projection(m_embed)
                        within_modality_loss += self.DGIloss(m_cls, m_embed, inverted_mask)
                    within_modality_loss /= self.n_modalities

                else:
                    N *= self.n_modalities

                    within_modality_loss += self.DGIloss(cls, nodes, inverted_mask.repeat(1,self.n_modalities))

                within_modality_loss *= self.args.DGI_lambda


            if self.args.do_NACL:
                cnt = 0
                if not self.args.specific:
                    nodes = nodes.reshape(B, self.n_modalities, N, -1)
                for i in range(self.n_modalities):
                    for j in range(self.n_modalities):
                        if i == j:
                            continue
                        cnt += 1
                        source = nodes[:, i, :, :]
                        target = nodes[:, j, :, :]

                        between_modality_loss += self.NACLloss(source, target, mask, mask, self.args.topk,
                                                               )
                between_modality_loss /= cnt
                between_modality_loss *= self.args.NACL_lambda

        if self.args.unimodal_inference:
            D = self.args.encoder_embed_dim

            if self.args.modalities == 'a':
                cols = slice(0, D)
            elif self.args.modalities == 't':
                cols = slice(D, 2 * D)
            elif self.args.modalities == 'v':
                cols = slice(2 * D, 3 * D)
            elif self.args.modalities == 'at':
                cols = slice(0, 2 * D)
            elif self.args.modalities == 'tv':
                cols = slice(D, 3 * D)
            elif self.args.modalities == 'av':
                w = self.linear_fusion[1].weight
                sliced_weight = torch.cat([w[:, 0:D], w[:, 2 * D:3 * D]], dim=-1)
                fused_emb = F.linear(fused_emb, sliced_weight, self.linear_fusion[1].bias)
            else:
                raise NotImplementedError(f"Unsupported modalities: {self.args.modalities}")

            if self.args.modalities != 'av':
                sliced_weight = self.linear_fusion[1].weight[:, cols]
                fused_emb = F.linear(fused_emb, sliced_weight, self.linear_fusion[1].bias)

        else:
            fused_emb = self.linear_fusion(fused_emb)
            # fused_emb = torch.stack(graphs, dim=-1)
            # graphs = fused_emb[inverted_mask, :]
            #
            # fused_emb = self.linear_fusion(fused_emb).squeeze()


        logits = self.classifier(fused_emb)[inverted_mask]
        labels = data['y'][inverted_mask].view(-1)

        if self.args.do_CE:
            class_weights = None

            if self.args.do_WCE:
                class_sample_count = torch.bincount(labels,
                minlength=self.args.num_classes).float()  # non-exist labels will be padded by zero
                class_sample_count[class_sample_count == 0] = float('inf')
                beta = 0.99
                eps = 1e-8
                class_weights = (1 - beta) / (1 - (beta ** class_sample_count) + eps)
                class_weights = torch.where(class_sample_count > 0, class_weights, torch.zeros_like(class_sample_count))

                class_weights /= class_weights.sum()
                # class_weights = torch.FloatTensor([1 / 0.089195,
                #                           1 / 0.144967,
                #                           1 / 0.165954,
                #                           1 / 0.160585,
                #                           1 / 0.111932,
                #                           1 / 0.262340]).to(logits.device)

                # proportions = torch.tensor([0.089195, 0.144967, 0.165954, 0.160585, 0.111932, 0.262340])
                #
                # # 1. 역수를 취해 기본 가중치 계산
                # weights = 1 / proportions
                #
                # # 2. 가중치를 정규화 (가중치의 합이 클래스 개수(6)가 되도록 스케일링)
                # normalized_weights = weights / torch.sum(weights) * len(proportions)
                #
                # class_weights = normalized_weights.to(logits.device)

            cross_entropy_loss = F.cross_entropy(logits, labels, weight=class_weights)

            # --- Unimodal Auxiliary CE Loss ---
            # graphs: (Valid_N, D, M)  →  graphs[:, :, i] gives (Valid_N, D) for modality i
            # BugFix: use per-modality graph embedding (not fused_emb), normalize OUTSIDE loop,
            #         multiply by unimodal_lambda (larger lambda = larger contribution, like NACL_lambda).
            # if train and self.n_modalities > 1:
            #     unimodal_ce_loss = 0.
            #     for i in range(self.n_modalities):
            #         uni_logits = self.unimodal_classifiers[i](fused_emb)[inverted_mask]
            #         unimodal_ce_loss += nn.functional.cross_entropy(uni_logits, labels, weight=class_weights)
            #     unimodal_ce_loss = (unimodal_ce_loss / self.n_modalities) * self.args.unimodal_lambda

        # Total Loss
        loss = cross_entropy_loss + within_modality_loss + between_modality_loss + unimodal_ce_loss
        # print(f"cross-entropy loss: {cross_entropy_loss}, within-modality loss: {within_modality_loss}, between-modality loss: {between_modality_loss}")

        return loss, logits, labels, graphs, fused_emb[inverted_mask, :], org_x[inverted_mask,:]
        
        
        
        # if self.args.dataset =="iemocap":
        #     data['utterance_order'] = torch.zeros_like(data['utterance_order'], dtype=torch.long, device=data['utterance_order'].device)
        # mask = data['mask'].clone()
        # sim_mask = mask.unsqueeze(1) | mask.unsqueeze(2)
        # data['mask'] = mask.repeat(1, self.n_modalities)
        # 
        # proj_list = []
        # 
        # if self.is_intra_dataset == False:
        #     if 'a' in self.modalities:
        #         proj_list.append(self.input_projection_a(data['audio']))
        #     if 't' in self.modalities:
        #         proj_list.append(self.input_projection_t(data['text']))
        #     if 'v' in self.modalities:
        #         proj_list.append(self.input_projection_v(data['visual']))
        # else:
        #     if 'a' in self.modalities:
        #         proj_list.append(self.model.input_projection_a(data['audio']))
        #     if 't' in self.modalities:
        #         proj_list.append(self.model.input_projection_t(data['text']))
        #     if 'v' in self.modalities:
        #         proj_list.append(self.model.input_projection_v(data['visual']))
        # 
        # B = data['y'].shape[0]
        # proj = torch.stack(proj_list, dim=1).view(B, -1, self.encoder_embed_dim)
        # data['x'] = self.layer_norm(proj)
        # 
        # representation = self.model.encoder(data, self.modalities)
        # 
        # embeddings = representation[:, 1:, :]  # real nodes (i.e., utterance tokens)
        # # summary = representation[:, 0, :]  # virtual nodes (i.e., graph tokens)
        # 
        # # Global-Local Mutual Information Maximization
        # B, N = embeddings.shape[:2]
        # fused_emb = None
        # 
        # multimodal_NCE_loss =0.
        # 
        # graphs = list()
        # for i in range(self.n_modalities):
        #     graphs.append(embeddings[:, i*n_max_utterances:(i+1)*n_max_utterances, :])
        # 
        # if self.args.do_NACL:
        #     for i, m_source in enumerate(graphs):
        #         for j, m_target in enumerate(graphs):
        #             if i == j:
        #                 continue
        #             multimodal_NCE_loss += self.NACLloss(m_source, m_target, sim_mask, self.args.topk, self.args.num_classes)
        #     multimodal_NCE_loss /= 6
        # 
        #         # Supervised (Cross Entropy) Loss
        # inverted_mask = mask.logical_not()
        # # fused_emb = torch.cat(graphs, dim=-1)
        # # logits = self.classifier(fused_emb)[inverted_mask].view(-1, self.args.num_classes)
        # 
        # # (M, B, N, D) -> (B, N, M, D) -> (B, N, MD)
        # # fused_emb = torch.stack(graphs).permute(1, 2, 0, 3).contiguous().view(B,n_max_utterances,-1)
        # fused_emb = torch.stack(graphs, dim=-1)
        # graphs = fused_emb[inverted_mask,:]
        # 
        # # === [OLD] Static Linear Fusion ===
        # # fused_emb_linear = self.linear_fusion(fused_emb).squeeze(-1)
        # # logits = self.classifier(self.activation(fused_emb_linear))[inverted_mask]
        # 
        # # === [NEW] Token-level Modality Attention ===
        # # 1. Reshape to (Batch, Nodes, Modalities, Dim)
        # fused_emb_permuted = fused_emb.permute(0, 1, 3, 2).contiguous() 
        # # 2. Calculate attention score for each modality token: (B, N, M, 1)
        # attention_logits = self.attention_fusion(fused_emb_permuted) 
        # # 3. Softmax over the Modality dimension (M)
        # attention_weights = torch.softmax(attention_logits, dim=2) 
        # # 4. Weighted sum of modalities: (B, N, D)
        # fused_emb_attn = (fused_emb_permuted * attention_weights).sum(dim=2) 
        # 
        # logits = self.classifier(self.activation(fused_emb_attn))[inverted_mask]
        # labels = data['y'][inverted_mask].view(-1)
        # 
        # if self.args.do_CE:
        #     class_weights = None
        #     if self.args.do_WCE:
        #         class_sample_count = torch.bincount(labels,
        #                                             minlength=self.args.num_classes).float()  # non-exist labels will be padded by zero
        #         # class_sample_count[class_sample_count == 0] = float('inf')
        #         # class_weights = 1.0 / class_sample_count  # non-exist labels' weights will be zero
        # 
        #         # beta = 0.99
        #         # eps = 1e-8
        #         # class_weights = (1 - beta) / (1 - (beta ** class_sample_count) + eps)
        #         # class_weights = torch.where(class_sample_count > 0, class_weights, torch.zeros_like(class_sample_count))
        #         #
        #         # class_weights /= class_weights.sum()
        #         # class_weights = torch.FloatTensor([1 / 0.086747,
        #         #                           1 / 0.144406,
        #         #                           1 / 0.227883,
        #         #                           1 / 0.160585,
        #         #                           1 / 0.127711,
        #         #                           1 / 0.252668]).to(logits.device)
        #         class_weights = torch.FloatTensor([1 / 0.167904,
        #                                   1 / 0.151342,
        #                                   1 / 0.169617,
        #                                   1 / 0.165620,
        #                                   1 / 0.173615,
        #                                   1 / 0.171902]).to(logits.device)
        # 
        # 
        #     cross_entropy_loss = nn.functional.cross_entropy(logits, labels, weight=class_weights)
        # 
        # # Total Loss
        # loss = cross_entropy_loss + multimodal_NCE_loss*self.args.multimodal_NACL_lambda
        # 
        # return loss, logits, labels, graphs
