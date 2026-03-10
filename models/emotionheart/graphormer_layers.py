# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self,
            hidden_dim,
            n_layers,
            num_nodes=None,
            num_degree=None,
            num_speakers=None,
            num_modalities=3,
    ):
        super(GraphNodeFeature, self).__init__()

        self.modality_encoder, self.order_encoder, self.speaker_encoder, self.in_degree_encoder, self.out_degree_encoder\
            = None, None, None, None, None
        #1 for graph token
        if num_modalities != 1:
            self.modality_encoder = nn.Embedding(num_modalities+1, hidden_dim, padding_idx = 0)
        if num_nodes != None:
            self.order_encoder = nn.Embedding(num_nodes+1, hidden_dim, padding_idx = 0)
        if num_speakers != None:
            self.speaker_encoder = nn.Embedding(num_speakers+1, hidden_dim, padding_idx = 0)
        if num_degree != None:
            self.in_degree_encoder = nn.Embedding(num_degree+1, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(
                num_degree+1, hidden_dim, padding_idx=0
            )


        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        x, in_degree, out_degree, utterance_order, speaker_identity, modality_position = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
            batched_data["utterance_order"],
            batched_data["speaker_identity"],
            batched_data["modality_position"]
        )
        n_graph, n_nodes = x.size()[0], x.size()[1]

        # node_feature = self.node_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]

        # if self.flag and perturb is not None:
        #     node_feature += perturb

        # node_feature = (
        #     node_feature
        #     + self.in_degree_encoder(in_degree)
        #     + self.out_degree_encoder(out_degree)
        # )

        # node_feature = x + in_degree.unsqueeze(dim=-1) + out_degree.unsqueeze(dim=-1)


        in_degree_feature = self.in_degree_encoder(in_degree) if self.in_degree_encoder is not None else 0.

        out_degree_feature = self.out_degree_encoder(out_degree) if self.out_degree_encoder is not None else 0.

        utterance_order_feature = self.order_encoder(utterance_order) if self.order_encoder is not None else 0.

        speaker_identity_feature = self.speaker_encoder(speaker_identity) if self.speaker_encoder is not None else 0.

        if modality_position is not None:
            modality_position_feature = self.modality_encoder(modality_position) if self.modality_encoder is not None else 0.
        else:
            modality_position_feature = 0.


        node_feature = x+in_degree_feature+out_degree_feature+utterance_order_feature+speaker_identity_feature+modality_position_feature
        # node_feature = x
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        num_heads,
        num_edges,
        num_spatial,
        num_edge_dis,
        edge_type,
        multi_hop_max_dist,
        n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = None
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )
        elif self.edge_type == "single_hop":
            self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        else:
            raise NotImplementedError(f"Unknown edge type: {self.edge_type}. Choose one of [multi_hop, single_hop]")

        self.spatial_pos_encoder = nn.Embedding(num_spatial + 1, num_heads, padding_idx=num_spatial)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x"],
        )
        # in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"],
        )

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # # spatial pos
        # # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        # spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).squeeze().permute(0, 3, 1, 2)
        # graph_attn_bias[:, :, 1:, 1:] += spatial_pos_bias

        # # reset spatial pos here
        # t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        # graph_attn_bias[:, :, 1:, 0] += t
        # graph_attn_bias[:, :, 0, :] += t
        #
        # # edge feature
        # if self.edge_type == "multi_hop":
        #     spatial_pos_ = spatial_pos.clone() # [n_graph, n_node, n_node, 1]
        #     spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1 for division
        #     # set 1 to 1, x > 1 to x - 1
        #     spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
        #     if self.multi_hop_max_dist > 0:
        #         spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
        #         edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
        #     # [n_graph, n_node, n_node, max_dist, n_head]
        #     edge_input = self.edge_encoder(edge_input).mean(-2)
        #     max_dist = edge_input.size(-2)
        #     edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
        #         max_dist, -1, self.num_heads
        #     ) # [max_dist, (n_graph, n_node, n_node), n_head]
        #     edge_input_flat = torch.bmm(
        #         edge_input_flat,
        #         self.edge_dis_encoder.weight.reshape(
        #             -1, self.num_heads, self.num_heads
        #         )[:max_dist, :, :],
        #     )
        #     edge_input = edge_input_flat.reshape(
        #         max_dist, n_graph, n_node, n_node, self.num_heads
        #     ).permute(1, 2, 3, 0, 4) # [n_graph, n_node, n_node, max_dist, n_head]
        #     edge_input = (
        #         edge_input.sum(-2) / spatial_pos_.float()
        #     ).permute(0, 3, 1, 2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        # else:
        #     # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        #     edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)
        #
        # graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        # graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias
