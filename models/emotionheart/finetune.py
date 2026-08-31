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

        # Keep only the pieces of the pretrained model that transfer: the encoder
        # (and, intra-dataset, the input projections); drop the pretraining decoder.
        if intra_dataset == False:
            data_embedding_dims = args.dataset_embedding_dims[args.dataset]
            if 'a' in args.modalities:
                self.input_projection_a = nn.Sequential(
                    nn.Linear(data_embedding_dims['a'], args.encoder_embed_dim),
                )

            if 't' in args.modalities:
                self.input_projection_t = nn.Sequential(
                    nn.Linear(data_embedding_dims['t'], args.encoder_embed_dim),
                )

            if 'v' in args.modalities:
                self.input_projection_v = nn.Sequential(
                    nn.Linear(data_embedding_dims['v'], args.encoder_embed_dim),
                )
            for name, module in list(self.model.named_children()):
                if not name.startswith("encoder"):
                    delattr(self.model, name)

        else:
            for name, module in list(self.model.named_children()):
                if not name.startswith("encoder") and not name.startswith("input_projection"):
                    delattr(self.model, name)

        if args.do_NACL == True:
            self.NACLloss = NACL_loss(args.temperature)
        if args.do_DGI == True:
            self.DGIloss = MIM_loss(args.encoder_embed_dim, args.temperature)
        if args.do_VATT == True:
            self.VATTloss = infoNCE_loss(args.temperature)

        self.linear_fusion = nn.Sequential(
            nn.LayerNorm(self.n_modalities * args.encoder_embed_dim),
            nn.Linear(self.n_modalities * args.encoder_embed_dim, args.encoder_embed_dim),
        )

        self.classifier = nn.Sequential(
            self.activation,
            nn.Dropout(args.dropout),
            nn.Linear(args.encoder_embed_dim, args.num_classes)
        )

    def freeze_encoder(self, param_name):
        """Freeze every parameter of the wrapped model whose name starts with `param_name`."""
        for name, param in self.model.named_parameters():
            if name.startswith(param_name):
                param.requires_grad = False
                print(f" Layer '{name}' is frozen.")


    def forward(self, data, train=False):
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

        proj_stack = torch.stack(proj_list, dim=1)  # [B, M, N, D]

        if self.args.unimodal_inference:
            # Missing-modality inference: zero out the projections of the dropped
            # modalities and mark their nodes as padding (True = masked out).
            keep = self.args.modalities  # subset of self.modalities, e.g. 'a', 'at', 'tv'
            if not set(keep) <= set(self.modalities):
                raise NotImplementedError(
                    f"Kept modalities '{keep}' must be a subset of the model's modalities '{self.modalities}'.")

            modal_masked = torch.zeros_like(proj_stack)
            mask_modality = torch.ones_like(mask, dtype=torch.bool)
            mask_parts = []
            for i, m in enumerate(self.modalities):
                if m in keep:
                    modal_masked[:, i] = proj_stack[:, i]
                    mask_parts.append(mask)
                else:
                    mask_parts.append(mask_modality)
            data['mask'] = torch.cat(mask_parts, dim=1)

            proj_stack = modal_masked

        else:
            data['mask'] = mask.repeat(1, self.n_modalities)

        org_x = proj_stack.permute(0, 2, 3, 1).contiguous()
        proj = proj_stack.reshape(B, -1, self.args.encoder_embed_dim)
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

        fused_emb = self.linear_fusion(fused_emb)

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

            cross_entropy_loss = F.cross_entropy(logits, labels, weight=class_weights)

        # Total Loss
        loss = cross_entropy_loss + within_modality_loss + between_modality_loss

        return loss, logits, labels, graphs, fused_emb[inverted_mask, :], org_x[inverted_mask,:]

