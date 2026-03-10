# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
)
from fairseq.utils import safe_hasattr

from .graphormer_graph_encoder import init_graphormer_params, GraphormerGraphEncoder
from .contrastive_loss import NACL_loss, MIM_loss, infoNCE_loss

logger = logging.getLogger(__name__)
logging.basicConfig(force=True, level=logging.INFO)


@register_model("graphormer")
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

class EmotionHeartModel(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        self.pretrained_encoder = None
        self.encoder_embed_dim = args.encoder_embed_dim
        self.modalities = args.modalities
        self.n_modalities = len(self.modalities)

        self.encoder = encoder

        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(args.encoder_embed_dim)

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
        if args.do_CLIP == True:
            # self.CLIP_projection = nn.Sequential(
            #     nn.LayerNorm(args.encoder_embed_dim),
            #     nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim*self.n_modalities),
            #     self.activation,
            #     nn.Linear(args.encoder_embed_dim*self.n_modalities, args.encoder_embed_dim)
            # )
            self.CLIPloss = infoNCE_loss(args.temperature)

        # self.linear_fusion = nn.Linear(self.n_modalities, 1)

        # concat multimodal graphormer
        self.linear_fusion = nn.Sequential(
            nn.LayerNorm(self.n_modalities * args.encoder_embed_dim),
            nn.Linear(self.n_modalities * args.encoder_embed_dim, args.encoder_embed_dim),
            self.activation,
        )

        # # concat unimodal multimodal graphormer
        # self.linear_fusion = nn.Sequential(
        #     nn.Linear(self.n_modalities * args.encoder_embed_dim*2, args.encoder_embed_dim),
        #     self.activation
        # )

        self.classifier = None
        if self.n_modalities == 1:
            self.classifier = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Linear(args.encoder_embed_dim , args.num_classes)
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
                nn.Dropout(args.dropout),
                # nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim // 4),
                # self.activation,
                # nn.Dropout(args.dropout),
                nn.Linear(args.encoder_embed_dim, args.num_classes)
            )

    def forward(self, data, n_max_utterances, train=False):
        if self.args.dataset == "iemocap":
            data['utterance_order'] = torch.zeros_like(data['utterance_order'], dtype=torch.long,
                                                       device=data['utterance_order'].device)

        within_modality_loss = 0.
        between_modality_loss = 0.
        cross_entropy_loss = 0.

        mask = data['mask'].clone()
        B, N = mask.shape
        inverted_mask = mask.logical_not()

        sim_mask = mask.unsqueeze(1) | mask.unsqueeze(2)
        data['mask'] = mask.repeat(1, self.n_modalities)

        proj_list = []
        if 'a' in self.modalities:
            proj_list.append(self.input_projection_a(data['audio']))
        if 't' in self.modalities:
            proj_list.append(self.input_projection_t(data['text']))
        if 'v' in self.modalities:
            proj_list.append(self.input_projection_v(data['visual']))

        org_x = torch.stack(proj_list, dim=1).permute(0,2,3,1).contiguous()
        proj = torch.stack(proj_list, dim=1).view(B, -1, self.args.encoder_embed_dim)
        data['x'] = self.layer_norm(proj)

        representation = self.encoder(data, self.modalities)

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
            if self.args.do_CLIP:
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

                        between_modality_loss += self.CLIPloss(source, target, sim_mask)
                between_modality_loss /= cnt
                between_modality_loss *= self.args.CLIP_lambda

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

                        between_modality_loss += self.NACLloss(source, target, sim_mask, self.args.topk,
                                                               self.args.num_classes)
                between_modality_loss /= cnt
                between_modality_loss *= self.args.NACL_lambda

        if self.args.unimodal_inference:
            if self.args.modalities == 'a':
                start_row = 0
                end_row = self.args.encoder_embed_dim
            elif self.args.modalities == 't':
                start_row = self.args.encoder_embed_dim
                end_row = self.args.encoder_embed_dim*2
            elif self.args.modalities == 'v':
                start_row = self.args.encoder_embed_dim*2
                end_row = self.args.encoder_embed_dim*3
            else:
                raise NotImplementedError
            sliced_weight = self.linear_fusion[1].weight[:, start_row:end_row]  # shape: (20, 300)
            sliced_bias = self.linear_fusion[1].bias # shape: (20,)
            fused_emb = F.linear(fused_emb, sliced_weight, sliced_bias)
            fused_emb = self.linear_fusion[2](fused_emb)
        else:
            fused_emb = self.linear_fusion(fused_emb)

        # if train:
        #     if self.args.do_DGI:
        #         within_modality_loss += self.DGIloss(fused_emb.mean(1), fused_emb, inverted_mask)*self.args.DGI_lambda

        # fused_emb = representation
        # b, u, e = fused_emb.shape
        # graphs = fused_emb.reshape(b,u,int(e//self.n_modalities),self.n_modalities)[inverted_mask, :]

        # embeddings = representation[:, 1:, :]  # real nodes (i.e., utterance tokens)
        # summary = representation[:, 0, :]  # virtual nodes (i.e., graph tokens)

        # Global-Local Mutual Information Maximization

        # B, N = embeddings.shape[:2]
        # fused_emb = None
        #
        # graphs = list()
        # for i in range(self.n_modalities):
        #     graphs.append(embeddings[:, i*n_max_utterances:(i+1)*n_max_utterances, :])
        #
        # if train:
        #     cnt = 0
        #     if self.args.do_NACL:
        #         for i, m_source in enumerate(graphs):
        #             for j, m_target in enumerate(graphs):
        #                 if i == j:
        #                     continue
        #                 cnt += 1
        #                 multimodal_NCE_loss += self.NCALloss(m_source, m_target, sim_mask, self.args.topk, self.args.num_classes)
        #         multimodal_NCE_loss /= cnt
        #
        #         # Supervised (Cross Entropy) Loss
        # # fused_emb = torch.cat(graphs, dim=-1)
        # # logits = self.classifier(fused_emb)[inverted_mask].view(-1, self.args.num_classes)
        #
        # # (M, B, N, D) -> (B, N, M, D) -> (B, N, MD)
        # fused_emb = torch.stack(graphs).permute(1, 2, 0, 3).contiguous().view(B,n_max_utterances,-1)
        # graphs = torch.stack(graphs, dim=-1)[inverted_mask, :]
        # fused_emb = self.linear_fusion(fused_emb)

        # fused_emb = torch.stack(graphs, dim=-1)
        # graphs = fused_emb[inverted_mask,:]
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
                class_weights = 1.0 / class_sample_count  # non-exist labels' weights will be zero

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

            cross_entropy_loss = 0.5*nn.functional.cross_entropy(logits, labels, weight=class_weights)

        # Total Loss
        loss = cross_entropy_loss + within_modality_loss + between_modality_loss
        # print(f"cross-entropy loss: {cross_entropy_loss}, within-modality loss: {within_modality_loss}, between-modality loss: {between_modality_loss}")

        return loss, logits, labels, graphs, fused_emb[inverted_mask, :], org_x[inverted_mask,:]


class EmotionHeartEncoder(FairseqEncoder):
    def __init__(self, args, n_nodes=None):
        super().__init__(dictionary=None)
        self.args = args
        self.n_modalities = len(args.modalities)

        num_nodes = args.n_max_utterances
        if n_nodes is not None:
            if num_nodes < n_nodes:
                num_nodes = n_nodes

        n_max_speakers = args.n_max_speakers
        if args.specific:
            self.modality_encoder = nn.ModuleDict()
            for m in args.modalities:
                self.modality_encoder[m] = GraphormerGraphEncoder(
                    # < for graphormer
                    num_nodes=num_nodes,
                    num_speakers=n_max_speakers,
                    num_degree=args.num_degree,
                    num_edges=args.num_edges,
                    num_modalities=1,
                    num_spatial=args.max_dist,
                    num_edge_dis=args.num_edge_dis,
                    edge_type=args.edge_type,
                    multi_hop_max_dist=args.multi_hop_max_dist,
                    # >
                    num_encoder_layers=args.encoder_layers,
                    embedding_dim=args.encoder_embed_dim,
                    ffn_embedding_dim=args.ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=args.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.act_dropout,
                    encoder_normalize_before=args.encoder_normalize_before,
                    pre_layernorm=args.pre_layernorm,
                    apply_graphormer_init=args.apply_graphormer_init,
                    activation_fn=args.activation_fn,
                )

        else:
            self.graph_encoder = GraphormerGraphEncoder(
                # < for graphormer
                num_nodes=None,#num_nodes,
                num_speakers=None,#n_max_speakers,
                num_degree=None,#args.num_degree,
                num_edges=args.num_edges,
                num_modalities=1,#self.n_modalities,
                num_spatial=args.max_dist,
                num_edge_dis=args.num_edge_dis,
                edge_type=args.edge_type,
                multi_hop_max_dist=args.multi_hop_max_dist,
                # >
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                encoder_normalize_before=args.encoder_normalize_before,
                pre_layernorm=args.pre_layernorm,
                apply_graphormer_init=args.apply_graphormer_init,
                activation_fn=args.activation_fn,
            )
        if args.hybrid:
            self.graph_encoder = GraphormerGraphEncoder(
                # < for graphormer
                num_nodes=None,
                num_speakers=None,
                num_degree=args.num_degree,
                num_edges=args.num_edges,
                num_modalities=self.n_modalities,
                num_spatial=args.max_dist,
                num_edge_dis=args.num_edge_dis,
                edge_type=args.edge_type,
                multi_hop_max_dist=args.multi_hop_max_dist,
                # >
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                encoder_normalize_before=args.encoder_normalize_before,
                pre_layernorm=args.pre_layernorm,
                apply_graphormer_init=args.apply_graphormer_init,
                activation_fn=args.activation_fn,
            )

        ## Concat multimodal graphormer
        # self.graph_encoder = GraphormerGraphEncoder(
        #     # < for graphormer
        #     num_nodes=num_nodes,
        #     num_speakers=n_max_speakers,
        #     num_degree=args.num_degree,
        #     num_edges=args.num_edges,
        #     num_modalities=self.n_modalities,
        #     num_spatial=args.max_dist,
        #     num_edge_dis=args.num_edge_dis,
        #     edge_type=args.edge_type,
        #     multi_hop_max_dist=args.multi_hop_max_dist,
        #     # >
        #     num_encoder_layers=args.encoder_layers,
        #     embedding_dim=args.encoder_embed_dim*self.n_modalities,
        #     ffn_embedding_dim=args.ffn_embed_dim,
        #     num_attention_heads=args.encoder_attention_heads,
        #     dropout=args.dropout,
        #     attention_dropout=args.attention_dropout,
        #     activation_dropout=args.act_dropout,
        #     encoder_normalize_before=args.encoder_normalize_before,
        #     pre_layernorm=args.pre_layernorm,
        #     apply_graphormer_init=args.apply_graphormer_init,
        #     activation_fn=args.activation_fn,
        # )

    def forward(self, batched_data, modality, perturb=None, masked_tokens=None, **unused):

        graphs = []
        nodes = []

        if self.n_modalities == 1:
            batched_data['modality_position'] = None

        if self.args.specific:
            for i, m in enumerate(modality):
                modality_batched_data = {}

                max_utterances = int(batched_data['x'].shape[1] // self.n_modalities)
                start = i * max_utterances
                end = (i + 1) * max_utterances

                modality_batched_data['x'] = batched_data['x'][:, start:end, :].clone()
                modality_batched_data['mask'] = batched_data['mask'][:, start:end].clone()
                modality_batched_data['utterance_order'] = batched_data['utterance_order'][:, start:end].clone()
                modality_batched_data['speaker_identity'] = batched_data['speaker_identity'][:, start:end].clone()

                in_degree_clone = batched_data['in_degree'][:, start:end].clone()
                # in_degree_clone[~modality_batched_data['mask']] -= (self.n_modalities - 1)
                modality_batched_data['in_degree'] = in_degree_clone

                out_degree_clone = batched_data['out_degree'][:, start:end].clone()
                # out_degree_clone[~modality_batched_data['mask']] -= (self.n_modalities - 1)
                modality_batched_data['out_degree'] = out_degree_clone

                new_attn_bias = torch.zeros(
                    (batched_data['attn_bias'].shape[0], max_utterances + 1, max_utterances + 1)).to(
                    batched_data['attn_bias'].device)
                new_attn_bias[:, 0, 0] = batched_data['attn_bias'][:, 0, 0].clone()
                new_attn_bias[:, 0, 1:] = batched_data['attn_bias'][:, 0, 1 + start:1 + end].clone()
                new_attn_bias[:, 1:, 0] = batched_data['attn_bias'][:, 1 + start:1 + end, 0].clone()
                new_attn_bias[:, 1:, 1:] = batched_data['attn_bias'][:, 1 + start:1 + end, 1 + start:1 + end].clone()
                modality_batched_data['attn_bias'] = new_attn_bias

                modality_batched_data['attn_edge_type'] = batched_data['attn_edge_type'][:, start:end, start:end,
                                                          :].clone()
                modality_batched_data['spatial_pos'] = batched_data['spatial_pos'][:, start:end, start:end, :].clone()
                modality_batched_data['edge_input'] = batched_data['edge_input'][:, start:end, start:end, :, :].clone()
                modality_batched_data['modality_position'] = None

                inner_states = self.modality_encoder[m](modality_batched_data, perturb=perturb,
                                                        n_modalities=self.n_modalities, use_attn_bias = True)
                inner_states = inner_states[-1].transpose(0, 1)
                graphs.append(inner_states[:, 0, :])
                nodes.append(inner_states[:, 1:, :])

            graphs = torch.stack(graphs, dim=1).unsqueeze(2)  # b, m, 1, e
            # graphs = graphs.mean(dim=1, keepdim=True)
            # nodes = torch.stack(nodes,dim=1)
            # b,_,_,e = nodes.shape
            # nodes = nodes.view(b, -1, e)

            # nodes = torch.stack(nodes, dim=2)
            # b, u, m, e = nodes.shape
            # nodes = nodes.view(b,u,-1)

            # u_nodes = torch.stack(nodes, dim=1)
            # b, m, u, e = u_nodes.shape
            # u_nodes = u_nodes.view(b,-1, e)

            nodes = torch.stack(nodes, dim=1)  # b, m, u, e
            b, _, _, e = nodes.shape

            z = torch.cat([graphs, nodes], dim=2)
            if self.args.hybrid:
                batched_data['x'] = nodes.reshape(b, -1, e)
                z = self.graph_encoder(batched_data, perturb=perturb, n_modalities=self.n_modalities)
                z = z[-1].transpose(0, 1)  # b, m*u, e
        else:
            z = self.graph_encoder(batched_data, perturb=perturb, n_modalities=self.n_modalities)
            z = z[-1].transpose(0, 1)  # b, m*u, e

        # if self.n_modalities == 1:
        #     batched_data['modality_position'] = None

        # z = torch.cat([graphs, nodes], dim=1)

        # batched_data['x'] = u_nodes
        #
        # m_nodes = self.graph_encoder(batched_data, perturb=perturb,n_modalities =self.n_modalities)
        # m_nodes = m_nodes[-1].transpose(0,1)[:,1:,:]
        #
        # z = torch.cat([u_nodes,m_nodes], dim=1).view(b,m*2,u,e).permute(0, 2, 1, 3).contiguous().view(b,u,-1)

        # modality_batched_data['x'] = nodes
        # z = self.graph_encoder(modality_batched_data, perturb=perturb,n_modalities =self.n_modalities)
        # z = z[-1].transpose(0, 1)

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        return z
