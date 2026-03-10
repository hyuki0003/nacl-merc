import numpy
import torch
from graphdata import algos
import math
import networkx as nx
import random
import copy
import numpy as np


# Function to compute eigenvector centrality with increased iterations and adjusted tolerance
def compute_eigenvector_centrality(G, max_iter=1000, tol=1e-6):
    try:
        centrality = nx.eigenvector_centrality_numpy(G, max_iter=max_iter, tol=tol)
    except nx.PowerIterationFailedConvergence:
        centrality = {node: 0.0 for node in G.nodes()}
    return centrality

def convert_to_single_emb(x, offset: int = 4):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


class iemocap_4_graphDataset():
    def __init__(self, dataset, name, args) -> None:
        self.args = args

        self._samples = copy.deepcopy(dataset)

        self.n_max_utterances = max(len(_s['speakers']) for _s in self._samples) if self._samples else 0
        self.n_max_speakers = max(len(set(_s['speakers'])) for _s in self._samples) if self._samples else 0
        self.modalities = args.modalities
        self.n_modalities = len(self.modalities)

        self.max_dist = args.max_dist

        self.dataset = args.dataset
        self.speaker_id_lookup_table = {"M": 1, "F": 2}
        self.embedding_dim = args.dataset_embedding_dims[args.dataset]

        self.num_samples = len(self._samples)
        self.num_batches = 1
        self.batch_size = self.num_samples
        if name == 'pretrain' or name=='finetune' or name=='train':
            self.batch_size = args.batch_size
            self.num_batches = math.ceil(self.num_samples / args.batch_size)

        self.batch_sizes = [self.batch_size]*(self.num_batches-1) + [self.num_samples%self.batch_size]\
            if self.num_samples%self.batch_size != 0 \
            else [self.batch_size]*(self.num_batches)

        self.samples = list()
        self.setup()
        self.batches = list()
        self.set_batch()

    def __getitem__(self, index):
        if self.samples == list():
            self.setup()
            self.set_batch()

        item = self.batches[int(index)]

        return item

    def __len__(self):
        return len(self.batches)

    def setup(self):
        for index, sample in enumerate(self._samples):
            data = {
                # Data
                "audio": None,
                "text" : None,
                "visual" : None,
                "x": None, # fused multimodal embeddings to be filled during training step
                "mask": None, # Identify virtural nodes (i.e., padded nodes)
                "y": None,

                # Additional Edge Information
                "attn_edge_type": None,
                "attn_bias": None,
                "spatial_pos": None,
                "edge_input": None,

                # Additional Node Information
                "modality_position": None,
                "utterance_order": None,
                "speaker_identity": None,
                "in_degree": None,
                "out_degree": None,
            }

            speaker_tensor = torch.Tensor(self._speakers_id(sample['speakers']))
            num_speakers = len(speaker_tensor)

            mask_tensor = torch.full((self.n_max_utterances,), False, dtype=torch.bool)
            mask_tensor[num_speakers:] = True

            modality_position = torch.zeros(self.n_max_utterances, dtype=torch.long).repeat(self.n_modalities)
            offset = torch.arange(1, self.n_modalities+1, dtype=torch.long).repeat_interleave(self.n_max_utterances)
            offset[mask_tensor.repeat(self.n_modalities)] = 0
            modality_position += offset

            if 'a' in self.modalities:
                modality_tensor = torch.zeros(self.n_max_utterances, self.embedding_dim['a']).to(torch.float32)
                padded_tensor = self._padding(sample['audio'], modality_tensor, dtype=torch.float32)
                data['audio'] = padded_tensor
            if 't' in self.modalities:
                modality_tensor = torch.zeros(self.n_max_utterances, self.embedding_dim['t']).to(torch.float32)
                padded_tensor = self._padding(sample['text'], modality_tensor, dtype=torch.float32)
                data['text'] = padded_tensor
            if 'v' in self.modalities:
                modality_tensor = torch.zeros(self.n_max_utterances, self.embedding_dim['v']).to(torch.float32)
                padded_tensor = self._padding(sample['visual'], modality_tensor, dtype=torch.float32)
                data['visual'] = padded_tensor


            # graph structures
            edge_index, edge_attr = self._set_relations(speaker_tensor)

            utterance_order, speaker_identity, in_degree, out_degree, attn_edge_type, attn_bias, spatial_pos, edge_input = self._preprocess_positions(speaker_tensor,edge_index,edge_attr,mask_tensor, node_centrality=self.args.centrality)

            label_tensor = torch.full((self.n_max_utterances,), -1, dtype= torch.long)
            label_tensor = self._padding(sample['labels'], label_tensor, dtype=torch.long)

            attn_edge_type_tensor = attn_edge_type.to(torch.long)
            attn_bias_tensor = attn_bias.to(torch.float32)
            spatial_pos_tensor = spatial_pos.to(torch.long)
            edge_input_tensor = edge_input.to(torch.long)
            in_degree_tensor = in_degree.to(torch.long)
            out_degree_tensor = out_degree.to(torch.long)

            data['mask'] = mask_tensor
            data['y'] = label_tensor
            data['modality_position'] = modality_position
            data['utterance_order'] = utterance_order
            data['speaker_identity'] = speaker_identity
            data['in_degree'] = in_degree_tensor
            data['out_degree'] = out_degree_tensor
            data['attn_edge_type'] = attn_edge_type_tensor
            data['attn_bias'] = attn_bias_tensor
            data['spatial_pos'] = spatial_pos_tensor
            data['edge_input'] = edge_input_tensor


            self.samples.append(data)

    def _padding(self, sample, tensor, dtype=torch.float32):
        cur_len = len(sample)
        if type(sample) == list or type(sample) == np.ndarray:
            tensor[:cur_len] = torch.tensor(sample, dtype=dtype)
        elif type(sample) == torch.Tensor:
            tensor[:cur_len] = sample.to(dtype=dtype)
        else:
            raise NotImplementedError('The type of the first argument "sample" should be a list or torch.Tensor')
        return tensor

    def _speakers_id(self, speakers):
        ids = []
        for s in speakers:
            ids.extend([self.speaker_id_lookup_table[s]])
        return ids

    def _unique_speakers(self, speakers):
        unique_ids = torch.unique(speakers)
        return unique_ids

    def _set_relations(self, speaker_tensor):
        """
        Relation 1 : Interlocuter Relationship,
        Relation 2 : Intralocuter Relationship,
        Relation 3 : Intermodality relation.
        """

        # Relation 1

        edge_index = torch.Tensor([])  # linked nodes i (1st row) and j (2nd row)
        edge_attr = torch.Tensor([])  # edge (relation) type

        num_speaker_nodes = len(speaker_tensor)  # except last node which cannot be source node, i.e., it can be only destination node
        if num_speaker_nodes >= 2:
            speaker_changes = torch.diff(speaker_tensor)
            base_nodes = torch.nonzero(speaker_changes).flatten()
            offsets = torch.arange(0, self.n_max_utterances * self.n_modalities,
                                   self.n_max_utterances).repeat_interleave(len(base_nodes))

            source_nodes = base_nodes.repeat(self.n_modalities) + offsets
            target_nodes = source_nodes + 1

            new_edge_index = torch.stack([source_nodes, target_nodes], dim=0)
            edge_index = torch.cat((edge_index, new_edge_index), dim=1)

            new_edge_attr = torch.full(size=(new_edge_index.shape[-1],), fill_value=0)
            edge_attr = torch.cat((edge_attr, new_edge_attr), dim=0)

        # Relation 2
        source_nodes, target_nodes = torch.Tensor([]), torch.Tensor([])

        unique_ids = self._unique_speakers(speaker_tensor)
        for id in unique_ids:
            subset_of_nodes = torch.where(speaker_tensor == id)[0]
            len_subset = subset_of_nodes.shape[0]

            if len_subset >= 2:
                source_nodes = torch.cat((source_nodes, subset_of_nodes[:-1]), dim=0)
                target_nodes = torch.cat((target_nodes, subset_of_nodes[1:]), dim=0)

        len_nodes = len(source_nodes)  # equal to the length of target node set

        offsets = torch.arange(0, self.n_max_utterances * self.n_modalities, self.n_max_utterances).repeat_interleave(
            len_nodes)

        source_nodes = torch.Tensor(source_nodes).repeat(self.n_modalities)
        target_nodes = torch.Tensor(target_nodes).repeat(self.n_modalities)

        source_nodes = source_nodes + offsets
        target_nodes = target_nodes + offsets

        new_edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        edge_index = torch.cat((edge_index, new_edge_index), dim=1)

        new_edge_attr = torch.full(size=(new_edge_index.shape[-1],), fill_value=1)
        edge_attr = torch.cat((edge_attr, new_edge_attr), dim=0)

        # Relation 3
        base_nodes = torch.arange(0, num_speaker_nodes).repeat(self.n_modalities)
        offsets = torch.arange(0, self.n_max_utterances * self.n_modalities, self.n_max_utterances).repeat_interleave(
            num_speaker_nodes)
        all_nodes = base_nodes + offsets

        reverse_offsets = torch.arange(self.n_max_utterances * (self.n_modalities - 1), -1,
                                       -self.n_max_utterances).repeat_interleave(num_speaker_nodes)
        reverse_nodes = base_nodes + reverse_offsets

        if self.n_modalities in ['at','tv','av']:
            source_nodes = torch.cat((all_nodes,reverse_nodes),dim=-1)
            target_nodes =torch.cat((reverse_nodes,all_nodes),dim=-1)

            new_edge_index = torch.stack([source_nodes, target_nodes], dim=0)
            edge_index = torch.cat((edge_index, new_edge_index), dim=1)

            new_edge_attr = torch.full(size=(new_edge_index.shape[-1],), fill_value=2)
            edge_attr = torch.cat((edge_attr, new_edge_attr), dim=0)

        elif self.modalities == 'atv':
            source_nodes = torch.cat([all_nodes[i * num_speaker_nodes:(i + 1) * num_speaker_nodes].repeat(self.n_modalities - 1)
                                for i in range(self.n_modalities)]) # A A T T V V
            target_nodes = torch.cat([all_nodes[j * num_speaker_nodes:(j + 1) * num_speaker_nodes] # T V A V A T
                                for i in range(self.n_modalities)
                                for j in range(self.n_modalities) if i != j])

            new_edge_index = torch.stack([source_nodes, target_nodes], dim=0)
            edge_index = torch.cat((edge_index, new_edge_index), dim=1)

            new_edge_attr = torch.full(size=(new_edge_index.shape[-1],), fill_value=2)
            edge_attr = torch.cat((edge_attr, new_edge_attr), dim=0)

        else:
            pass

        # # Relation 4
        # new_edge_index = torch.stack([all_nodes, all_nodes], dim=0)  # self connections
        # edge_index = torch.cat((edge_index, new_edge_index), dim=1)
        #
        # new_edge_attr = torch.full(size=(new_edge_index.shape[-1],), fill_value=3)
        # edge_attr = torch.cat((edge_attr, new_edge_attr), dim=0)

        return edge_index.to(torch.long), edge_attr.to(torch.long)

    def _eigen_centrality(self, adj):
        # Convert the adjacency matrix to a NetworkX directed graph
        G = nx.DiGraph()

        # Add edges to the graph based on non-zero entries in the adjacency matrix
        rows, cols = adj.nonzero(as_tuple=True)
        edges = zip(rows.tolist(), cols.tolist())
        G.add_edges_from(edges)

        # Ensure all nodes (0 to 155) are included in the graph
        for i in range(adj.shape[0]):
            if i not in G:
                G.add_node(i)

        # Compute out-eigenvector centrality
        out_degree = compute_eigenvector_centrality(G)
        out_degree_tensor = torch.zeros(adj.shape[0], dtype=torch.float)
        for node in range(adj.shape[0]):
            out_degree_tensor[node] = out_degree.get(node, 0.0)

        # Compute in-eigenvector centrality by reversing the graph
        G_reversed = G.reverse()
        in_degree = compute_eigenvector_centrality(G_reversed)
        in_degree_tensor = torch.zeros(adj.shape[0], dtype=torch.float)
        for node in range(adj.shape[0]):
            in_degree_tensor[node] = in_degree.get(node, 0.0)

        return in_degree_tensor, out_degree_tensor

    def _preprocess_positions(self, speakers, edge_index, edge_attr, mask, node_centrality='degree'):
        num_nodes = self.n_max_utterances*self.n_modalities
        num_speaker_nodes = len(speakers)

        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        # By transposing, we set the adjacency matrix to indicate the correct direction
        # from the perspective of Feature propagation and aggregation strategy of GNNs.
        adj = adj.T # So now rows represent destination, while columns represent source nodes.

        # For additional node information
        utterance_order = torch.zeros((self.n_max_utterances,), dtype=torch.long)
        utterance_order[:num_speaker_nodes] = torch.arange(1,num_speaker_nodes+1, dtype=torch.long)
        utterance_order = utterance_order.repeat(self.n_modalities)

        speaker_identity = torch.zeros((self.n_max_utterances,), dtype=torch.long)
        speaker_identity[:num_speaker_nodes] = speakers
        speaker_identity = speaker_identity.repeat(self.n_modalities)

        in_degree_tensor, out_degree_tensor = None, None

        if node_centrality == 'degree':
            in_degree_tensor = adj.long().sum(dim=1).view(-1) # degree
            out_degree_tensor = adj.sum(dim=0).view(-1) # degree

        elif node_centrality == 'eigen':
            in_degree_tensor, out_degree_tensor = self._eigen_centrality(adj)

        else:
            raise NotImplementedError(f"Unavailable Centrality Type: {node_centrality},"
                                      f" Choose one between \'degree\' or \'eigen\' ")


        # For additional edge information
        attn_edge_type = torch.zeros((num_nodes,num_nodes), dtype= torch.long)
        attn_edge_type[edge_index[0,:], edge_index[1,:]] = convert_to_single_emb(edge_attr) + 1 # add 1 to distinguish zero components
        attn_edge_type = attn_edge_type.T
        attn_edge_type = attn_edge_type.unsqueeze(dim = -1)

        shortest_path_result, path = algos.floyd_warshall(adj, max_val = self.max_dist)
        edge_input = algos.gen_edge_input(self.max_dist, path, attn_edge_type).to(torch.long)

        # All nodes pass themselves before going others
        spatial_pos_preprocess = shortest_path_result.clone()
        mask_repeat_sim = mask.repeat(self.n_modalities)
        masked_identity_matrix = mask_repeat_sim.diag()
        spatial_pos_preprocess[masked_identity_matrix] = self.max_dist # add unreachable value to padded nodes' self-loop elements

        spatial_pos = spatial_pos_preprocess.to(torch.long).unsqueeze(dim=-1)

        num_nodes_added_CLS = num_nodes + 1
        attn_bias = torch.zeros([num_nodes_added_CLS, num_nodes_added_CLS], dtype=torch.float)

        return utterance_order, speaker_identity, in_degree_tensor, out_degree_tensor, attn_edge_type, attn_bias, spatial_pos, edge_input


    def shuffle(self):
        random.shuffle(self.samples)
        return

    def set_batch(self):
        self.shuffle()
        self.batches = list()

        sample = self.samples[0]

        idx = 0
        audio, text, visual = None, None, None
        for size in self.batch_sizes:
            group = self.samples[idx:idx+size]
            idx += size

            if 'a' in self.modalities:
                audio = torch.zeros_like(sample['audio']).repeat(size, 1, 1)
            if 't' in self.modalities:
                text = torch.zeros_like(sample['text']).repeat(size, 1, 1)
            if 'v' in self.modalities:
                visual = torch.zeros_like(sample['visual']).repeat(size, 1, 1)

            mask = torch.full_like(sample['mask'], False).repeat(size, 1)
            y = torch.full_like(sample['y'], fill_value= -1).repeat(size, 1)
            modality_position = torch.zeros_like(sample['modality_position']).repeat(size, 1)
            utterance_order = torch.zeros_like(sample['utterance_order']).repeat(size, 1)
            speaker_identity = torch.zeros_like(sample['speaker_identity']).repeat(size, 1)
            in_degree = torch.zeros_like(sample['in_degree']).repeat(size, 1)
            out_degree = torch.zeros_like(sample['out_degree']).repeat(size, 1)
            attn_edge_type = torch.zeros_like(sample['attn_edge_type']).repeat(size, 1, 1, 1)
            attn_bias = torch.zeros_like(sample['attn_bias']).repeat(size, 1, 1)
            spatial_pos = torch.zeros_like(sample['spatial_pos']).repeat(size, 1, 1, 1)
            edge_input = torch.zeros_like(sample['edge_input']).repeat(size, 1, 1, 1, 1)

            batched_data = {
                "audio": None,
                "text": None,
                "visual": None,
                # "audio_mask": None,
                # "text_mask": None,
                # "visual_mask": None,
                "x": None,  # fused multimodal embeddings to be filled during training step
                "mask": None,
                "y": None,
                "modality_position": None,
                "utterance_order": None,
                "speaker_identity": None,
                "in_degree": None,
                "out_degree": None,
                "attn_edge_type": None,
                "attn_bias": None,
                "spatial_pos": None,
                "edge_input": None
            }

            for i, g in enumerate(group):
                if 'a' in self.modalities:
                    audio[i] = g['audio']
                if 't' in self.modalities:
                    text[i] = g['text']
                if 'v' in self.modalities:
                    visual[i] = g['visual']
                mask[i] = g['mask']
                y[i] = g['y']
                modality_position[i] = g['modality_position']
                utterance_order[i] = g['utterance_order']
                speaker_identity[i] = g['speaker_identity']
                attn_edge_type[i] = g['attn_edge_type']
                attn_bias[i] = g['attn_bias']
                spatial_pos[i] = g['spatial_pos']
                edge_input[i] = g['edge_input']
                in_degree[i] = g['in_degree']
                out_degree[i] = g['out_degree']

            batched_data['audio'] = audio
            batched_data['text'] = text
            batched_data['visual'] = visual
            batched_data['mask'] = mask
            batched_data['y'] = y
            batched_data['modality_position'] = modality_position
            batched_data['utterance_order'] = utterance_order
            batched_data['speaker_identity'] = speaker_identity
            batched_data['in_degree'] = in_degree
            batched_data['out_degree'] = out_degree
            batched_data['attn_edge_type'] = attn_edge_type
            batched_data['attn_bias'] = attn_bias
            batched_data['spatial_pos'] = spatial_pos
            batched_data['edge_input'] = edge_input

            self.batches.append(batched_data)

        return