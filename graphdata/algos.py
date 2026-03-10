import torch
import numpy as np

def floyd_warshall(adjacency_matrix, max_val=51):
    n = adjacency_matrix.shape[0]
    assert adjacency_matrix.shape[1] == n, "Adjacency matrix must be square"

    # Convert adjacency matrix to float32 and replace zeros with max_val (except diagonal)
    M = adjacency_matrix.clone().float()
    M[M == 0] = max_val
    torch.diagonal(M).fill_(0)  # Set diagonal to 0

    # Initialize the path matrix
    path = -1 * torch.ones((n, n), dtype=torch.int64)
    direct_connections = (M != max_val) & (torch.arange(n).unsqueeze(1) != torch.arange(n).unsqueeze(0))
    path[direct_connections] = torch.arange(n).unsqueeze(0).repeat(n, 1)[direct_connections]

    # Floyd-Warshall algorithm with vectorized operations
    for k in range(n):
        # Expand k-th row and k-th column to compare all pairs simultaneously
        M_new = torch.minimum(M, M[:, k].unsqueeze(1) + M[k, :].unsqueeze(0))
        update_mask = M > M_new  # Identify pairs where the new path is shorter

        # Protect direct connections
        direct_connection_mask = direct_connections.clone()

        # Update path matrix to reflect the new intermediate node
        intermediate_nodes = path[:, k].unsqueeze(1).repeat(1, n)  # Expand path[k, j]
        path[update_mask] = torch.where(
            direct_connection_mask[update_mask],  # Directly connected paths are protected
            path[update_mask],  # Keep the current path
            torch.where(
                intermediate_nodes[update_mask] != -1,
                intermediate_nodes[update_mask],
                k
            )
        )

        M = M_new

    # Set unreachable paths
    M[M >= max_val] = max_val
    path[M == max_val] = -1

    return M, path

#
def get_all_edges_iterative(path, max_dist):
    """
    Reconstruct all shortest paths using the path matrix in a vectorized manner.
    """
    # n = path.shape[0]
    # all_paths = -torch.ones((n, n, max_dist + 1), dtype=torch.int64, device=path.device)
    #
    # # Initialize first step (starting nodes)
    # all_paths[:, :, 0] = torch.arange(n).view(1, -1).repeat(n, 1)
    #
    # # Iteratively fill paths up to max_dist
    # for k in range(1, max_dist + 1):
    #     previous_nodes = all_paths[:, :, k - 1]  # Nodes from the previous step
    #     valid_previous = previous_nodes != -1  # Check if previous step was valid
    #
    #     # Set invalid entries to 0 temporarily for gather (will not be updated due to mask)
    #     previous_nodes_safe = previous_nodes.clone()
    #     previous_nodes_safe[~valid_previous] = 0  # Replace -1 with valid index for safe gather
    #
    #     # Gather next nodes based on valid previous
    #     next_nodes = torch.gather(path, 1, previous_nodes_safe)
    #     next_nodes[~valid_previous] = -1
    #     # Create valid mask for current step
    #     valid_mask = valid_previous & (next_nodes != -1)
    #     rows, cols = valid_mask.nonzero(as_tuple=True)
    #
    #     # Update paths using valid indices
    #     all_paths[rows, cols, k] = next_nodes[rows, cols]
    #
    # return all_paths

    num_nodes = path.shape[0]

    # Initialize all_paths tensor with -1 (indicating no node initially)
    all_paths = -torch.ones((num_nodes, num_nodes, max_dist+1), dtype=torch.long)

    # Path from a node to itself is just the node
    all_paths[:, :, 0] = torch.arange(num_nodes).unsqueeze(1).expand(-1, num_nodes)

    # Create a tensor to track the current nodes being processed
    current_nodes = all_paths[:, :, 0].clone()

    for k in range(1, max_dist +1):
        # Get the next nodes using path_matrix
        next_nodes = path[current_nodes, torch.arange(num_nodes).unsqueeze(0).expand(num_nodes, -1)]

        # Identify valid updates where there is a valid next node
        valid_updates = (next_nodes != -1)
        rows, cols = valid_updates.nonzero(as_tuple=True)
        # Update all_paths and current_nodes for valid paths
        all_paths[rows,cols, k] = next_nodes[rows,cols]
        current_nodes[rows,cols] = next_nodes[rows,cols]

        # Break early if no updates are made
        if not valid_updates.any():
            break

    return all_paths

#
def gen_edge_input(max_dist, path, edge_feat):
    """
    Generate edge input features for all pairs in a fully vectorized manner.
    """
    n = path.shape[0]
    assert path.shape[1] == n, "Path matrix must be square"

    # Ensure edge_feat is a tensor
    if isinstance(edge_feat, np.ndarray):
        edge_feat = torch.from_numpy(edge_feat).to(path.device)

    # Initialize edge features tensor
    edge_fea_all = torch.zeros((n, n, max_dist, edge_feat.shape[-1]), dtype=torch.int64, device=path.device)

    # Reconstruct paths for all node pairs
    all_paths = get_all_edges_iterative(path, max_dist)

    # Assign edge features for valid paths
    for k in range(max_dist):
        src = all_paths[:, :, k]
        dst = all_paths[:, :, k + 1]

        # Ensure both src and dst are valid
        valid_edges = (src != -1) & (dst != -1)
        rows, cols = valid_edges.nonzero(as_tuple=True)

        # Extract valid indices for src and dst
        src_valid = src[rows, cols]
        dst_valid = dst[rows, cols]

        # Check for index validity
        assert torch.all((src_valid >= 0) & (src_valid < edge_feat.shape[0])), "Invalid src index"
        assert torch.all((dst_valid >= 0) & (dst_valid < edge_feat.shape[1])), "Invalid dst index"

        # Assign edge features and verify correctness
        assigned_features = edge_feat[src_valid, dst_valid, :]

        # Swap rows and cols for reversed i (source) and j (destination)
        edge_fea_all[rows, cols, k, :] = assigned_features  # i and j are swapped here

    return edge_fea_all

