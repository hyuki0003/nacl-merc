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
from .contrastive_loss import NACL_loss, infoNCE_loss  # , MILNACLloss
from .multihead_attention import MultiheadAttention

logger = logging.getLogger(__name__)
logging.basicConfig(force=True, level=logging.INFO)

@register_model("graphormer")
class EmotionHeartModel(FairseqEncoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder)
        self.args = args
        self.x = 0
        if getattr(args, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)
        self.encoder_embed_dim = args.encoder_embed_dim
        self.modalities = args.modalities
        self.n_modalities = len(self.modalities)
        self.activation = nn.GELU()
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

        self.layer_norm = LayerNorm(args.encoder_embed_dim)
        # Learnable mask token (N(0, 0.02) 정규 분포 초기화, truncation [-2, 2])

        self.mask_token = None
        self.decoder = None
        if args.do_MAE:
            self.mask_token = nn.Parameter(torch.empty(self.encoder_embed_dim))
            nn.init.trunc_normal_(self.mask_token, mean=0, std=0.02, a=-2, b=2)
            self.decoder = decoder

        self.NACLloss = None
        if args.do_NACL:
            self.NACLloss = NACL_loss(args.temperature)
        elif args.do_VATT:
            self.NACLloss = infoNCE_loss(args.temperature)


    def mask(self, tensor, pad_mask, mask_ratio=0.15):

        batch_size, seq_len, emb_dim = tensor.shape[:3]

        valid_mask = ~pad_mask

        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=tensor.device)
        for i in range(batch_size):
            valid_indices = torch.nonzero(valid_mask[i], as_tuple=True)[0]
            len_valid_indices = len(valid_indices)
            if len_valid_indices <= 4:
                continue

            mask_count = int(mask_ratio*len_valid_indices)
            masked_indices = valid_indices[torch.randperm(len(valid_indices))[:mask_count]] # 랜덤 선택
            mask[i, masked_indices] = True

        return mask

    def forward(self, batched_data, modal, **kwargs):
        return self.encoder(batched_data, modal, **kwargs)

    def pretrain(self, data, n_max_utterances):
        padding_mask = data['mask'].clone()

        encoder_output_all = torch.zeros(self.args.batch_size, n_max_utterances * self.n_modalities,
                                         self.args.encoder_embed_dim, dtype=torch.float32,
                                         device=data['mask'].device)

        masked_tokens = {} # ground-truth
        original_entities = {}
        padding_mask_all = {}
        token_mask_all = {}
        proj_list = []
        encoded_all_tokens = {}
        if 'a' in self.modalities:
            mask_a = self.mask(data['audio'], padding_mask, mask_ratio=self.args.mask_prob_a)
            padding_mask_a = padding_mask.clone()
            padding_mask_a[mask_a] = True
            padding_mask_all['a'] = padding_mask_a
            token_mask_all['a'] = mask_a
            masked_tokens['a'] = data['audio'].clone()[mask_a]
            data['audio'][mask_a]= 0.
            proj_list.append(self.input_projection_a(data['audio']))
        if 't' in self.modalities:
            mask_t = self.mask(data['text'], padding_mask, mask_ratio=self.args.mask_prob_t)
            padding_mask_t = padding_mask.clone()
            padding_mask_t[mask_t] = True
            padding_mask_all['t'] = padding_mask_t
            token_mask_all['t'] = mask_t
            masked_tokens['t'] = data['text'].clone()[mask_t]
            data['text'][mask_t]=0.
            proj_list.append(self.input_projection_t(data['text']))
        if 'v' in self.modalities:
            mask_v = self.mask(data['visual'], padding_mask, mask_ratio=self.args.mask_prob_v)
            padding_mask_v = padding_mask.clone()
            padding_mask_v[mask_v] = True
            padding_mask_all['v'] = padding_mask_v
            token_mask_all['v'] = mask_v
            masked_tokens['v'] = data['visual'].clone()[mask_v]
            data['visual'][mask_v]=0.
            proj_list.append(self.input_projection_v(data['visual']))

        B, NM, _ = encoder_output_all.shape # (B, N*M)

        data['mask'] = torch.stack(list(padding_mask_all.values()), dim=1).view (B,-1)
        mask_token_mask = torch.stack(list(token_mask_all.values()), dim=1).view(B, -1) # (B, N*M)
        mask_token_mask_expand = mask_token_mask.unsqueeze(1) | mask_token_mask.unsqueeze(2) # (B, N*M, N*M)
        attn_bias_mask = torch.zeros((B, NM+1, NM+1), device=mask_token_mask.device, dtype=mask_token_mask.dtype)
        attn_bias_mask[:, 1:, 1:] = mask_token_mask_expand

        proj = torch.stack(proj_list, dim=1).view(B, -1, self.args.encoder_embed_dim)
        data['x'] = self.layer_norm(proj)


        original_entities['modality_position'] = data['modality_position'].clone()
        original_entities['utterance_order'] = data['utterance_order'].clone()
        original_entities['speaker_identity'] = data['speaker_identity'].clone()
        original_entities['in_degree'] = data['in_degree'].clone()
        original_entities['out_degree'] = data['out_degree'].clone()
        data['modality_position'][mask_token_mask] = 0.
        data['utterance_order'][mask_token_mask] = 0.
        data['speaker_identity'][mask_token_mask] = 0.
        data['in_degree'][mask_token_mask] = 0.
        data['out_degree'][mask_token_mask] = 0.
        data['attn_edge_type'][mask_token_mask_expand, :] = 0.
        data['attn_bias'][attn_bias_mask] = 0.
        data['spatial_pos'][mask_token_mask_expand, :] = self.args.max_dist
        data['edge_input'][mask_token_mask_expand,:,:] = 0


        representation = self.forward(data, self.args.modalities)

        embeddings = representation[:, 1:, :]  # real nodes (i.e., utterance tokens)
        #summary = representation[:, 0, :]  # virtual nodes (i.e., graph tokens)

        valid_token_mask = ~data['mask']
        encoder_output_all[mask_token_mask] = self.mask_token
        encoder_output_all[valid_token_mask] = embeddings[valid_token_mask]

        original_entities['x'] = encoder_output_all

        encoder_output_all = self.encoder.graph_encoder.graph_node_feature(original_entities)[:,1:,:]
        encoder_output_all = self.layer_norm(encoder_output_all)

        modality_ratio = {
            'a': self.args.mask_prob_a,
            't': self.args.mask_prob_t,
            'v': self.args.mask_prob_v
        }
        total_ratio = sum(modality_ratio.values())  # 0.5 + 0.3 + 0.7 = 1.5
        normalized_loss_weights = {modality: ratio / total_ratio for modality, ratio in modality_ratio.items()}

        for i, m in enumerate(self.modalities):
            encoded_all_tokens[m] = encoder_output_all[:,i*n_max_utterances:(i+1)*n_max_utterances,:]

        #

        # Loss 1
        masked_neighbor_aligned_contrastive_loss = 0.
        if self.args.do_NACL:
            scaler = 0.5 if self.n_modalities == 3 else 1
            for m_source in self.modalities:
                for m_target in self.modalities:
                    if m_source == m_target:
                        continue

                    source_all_mask = padding_mask_all[m_source]
                    source_all_mask_sim = source_all_mask.unsqueeze(2) | source_all_mask.unsqueeze(1)

                    masked_neighbor_aligned_contrastive_loss += (
                                scaler * normalized_loss_weights[m_source] * self.NACLloss(encoded_all_tokens[m_source],
                                                                                          encoded_all_tokens[m_target],
                                                                                          source_all_mask_sim,
                                                                                          self.args.topk,
                                                                                          self.args.num_classes))

        # self.x += 1
        # if self.x == 10:
        #     print("stop ", self.x)

        # Loss 2
        MAE_reconstruction_loss = 0.
        for modality, masked_token in masked_tokens.items():
            m_token_mask = token_mask_all[modality]
            m_token_mask_all = torch.zeros_like(data['mask'])
            m_token_mask_all[:, :m_token_mask.shape[1]] = m_token_mask
            m_token_mask_all = ~m_token_mask_all

            query = encoder_output_all.clone()

            query[m_token_mask_all] = 0.
            key_value = encoder_output_all.clone()
            key_value[data['mask']] = 0.

            attn_mask = m_token_mask_all.unsqueeze(2) | data['mask'].unsqueeze(1)

            # if modality=='v' and self.x==10:
            #     print("stop")
            reconstructed_token = self.decoder.forward(query, key_value, attn_mask, modality)
            reconstructed_token = reconstructed_token[~m_token_mask_all]

            MAE_reconstruction_loss += (F.mse_loss(reconstructed_token, masked_token)*normalized_loss_weights[modality])

        # print(f"MAE loss : {MAE_reconstruction_loss}")
        loss = self.args.multimodal_NACL_lambda*masked_neighbor_aligned_contrastive_loss + self.args.multimodal_MAE_lambda*MAE_reconstruction_loss
        return loss

    def pretrain_NACL(self, data, n_max_utterances):
        mask = data['mask'].clone()
        data['mask'] = mask.repeat(1,self.n_modalities)
        proj_list = []
        if 'a' in self.modalities:
            proj_list.append(self.input_projection_a(data['audio']))
        if 't' in self.modalities:
            proj_list.append(self.input_projection_t(data['text']))
        if 'v' in self.modalities:
            proj_list.append(self.input_projection_v(data['visual']))
        B = data['audio'].shape[0]

        # Ensure there's at least one modality before stacking/concatenating

        proj = torch.stack(proj_list, dim=1).view(B, -1, self.args.encoder_embed_dim)

        data['x'] = self.layer_norm(proj)

        representation = self.forward(data, self.args.modalities)

        embeddings = representation[:, 1:, :]  # real nodes (i.e., utterance tokens)
        # summary = representation[:, 0, :]  # virtual nodes (i.e., graph tokens)

        encoded_all_tokens = {}
        for i, m in enumerate(self.modalities):
            encoded_all_tokens[m] = embeddings[:,i*n_max_utterances:(i+1)*n_max_utterances,:]

        masked_neighbor_aligned_contrastive_loss = 0.
        scaler = 1.
        if self.n_modalities > 1:
            scaler = 1/(self.n_modalities+(self.n_modalities-1))
        if self.args.do_NACL:
            padding_mask_sim = mask.unsqueeze(2) | mask.unsqueeze(1)
            for m_source in self.modalities:
                for m_target in self.modalities:
                    if m_source == m_target:
                        continue
                    masked_neighbor_aligned_contrastive_loss += (
                            scaler * self.NACLloss(encoded_all_tokens[m_source],
                                                                                          encoded_all_tokens[m_target],
                                                                                          padding_mask_sim,
                                                                                          self.args.topk,
                                                                                          self.args.num_classes))

        return masked_neighbor_aligned_contrastive_loss

    def pretrain_VATT(self, data, n_max_utterances):
        mask = data['mask'].clone()
        data['mask'] = mask.repeat(1,self.n_modalities)
        proj_list = []
        if 'a' in self.modalities:
            proj_list.append(self.input_projection_a(data['audio']))
        if 't' in self.modalities:
            proj_list.append(self.input_projection_t(data['text']))
        if 'v' in self.modalities:
            proj_list.append(self.input_projection_v(data['visual']))
        B = data['audio'].shape[0]

        # Ensure there's at least one modality before stacking/concatenating

        proj = torch.stack(proj_list, dim=1).view(B, -1, self.args.encoder_embed_dim)

        data['x'] = self.layer_norm(proj)

        representation = self.forward(data, self.args.modalities)

        embeddings = representation[:, 1:, :]  # real nodes (i.e., utterance tokens)
        # summary = representation[:, 0, :]  # virtual nodes (i.e., graph tokens)

        encoded_all_tokens = {}
        for i, m in enumerate(self.modalities):
            encoded_all_tokens[m] = embeddings[:,i*n_max_utterances:(i+1)*n_max_utterances,:]

        VATT_loss = 0.
        scaler = 1.
        if self.n_modalities > 1:
            scaler = 1/(self.n_modalities+(self.n_modalities-1))
        if self.args.do_VATT:
            padding_mask_sim = mask.unsqueeze(2) | mask.unsqueeze(1)
            for m_source in self.modalities:
                for m_target in self.modalities:
                    if m_source == m_target:
                        continue
                    VATT_loss += (
                            scaler * self.NACLloss(encoded_all_tokens[m_source],
                                                  encoded_all_tokens[m_target],
                                                  padding_mask_sim,
                                                  )
                            )

        return VATT_loss


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



class EmotionHeartDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Cross-Attention Layer
        self.n_encoder_attention_heads = args.encoder_attention_heads
        self.cross_attention = MultiheadAttention(embed_dim=args.encoder_embed_dim, num_heads=self.n_encoder_attention_heads)
        self.norm = nn.LayerNorm(args.encoder_embed_dim)  # LayerNorm after Cross-Attention

        # Feed Forward Network (FFN)
        data_embedding_dims = args.dataset_embedding_dims[args.dataset]

        self.ffn = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim * 4),
            nn.GELU(),  # GELU 활성화 함수
            nn.Linear(args.encoder_embed_dim * 4, args.encoder_embed_dim)
        )

        self.proj = nn.ModuleDict()
        for m in args.modalities:
            self.proj[m] = nn.Linear(args.encoder_embed_dim, data_embedding_dims[m])


    def forward(self, decoder_input, cross_modal_context,attn_mask, modality):
        attn_mask = attn_mask.float()
        attn_mask.masked_fill_(attn_mask == 1., float('-inf'))
        attn_mask = attn_mask.repeat(self.n_encoder_attention_heads, 1, 1)

        query = decoder_input.transpose(0, 1)
        key, value = cross_modal_context.transpose(0,1), cross_modal_context.transpose(0, 1)
        x, _ = self.cross_attention(query, key, value, attn_mask=attn_mask)
        x = self.norm(query + x)

        res = x
        x = self.ffn(x)
        x = self.norm(x + res)

        x = self.proj[modality](x)

        x = x.transpose(0,1)

        return x
