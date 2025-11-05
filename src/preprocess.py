# File: src/preprocess.py
import pandas as pd
import numpy as np
import networkx as nx
import torch
from scipy.sparse import csr_matrix, hstack, vstack  # <-- import functions directly


def create_graph(df: pd.DataFrame):
    """
    Build a transaction graph with safe padding for isolated nodes.
    Returns:
        adj_tensor : torch.sparse_coo_tensor (N, N) coalesced
        node_order : original index order (not used)
    """
    G = nx.Graph()
    df_sorted = df.sort_values('Time').reset_index(drop=True)

    # --------------------------------------------------------------
    # 1. Add edges (time < 1h and amount diff < $50)
    # --------------------------------------------------------------
    edges = []
    for i in range(len(df_sorted) - 1):
        r1 = df_sorted.iloc[i]
        r2 = df_sorted.iloc[i + 1]
        if (r2['Time'] - r1['Time']) < 3600 and abs(r2['Amount'] - r1['Amount']) < 50:
            edges.append((i, i + 1))
    G.add_edges_from(edges)

    n_total = len(df)

    # --------------------------------------------------------------
    # 2. If no edges → return zero matrix
    # --------------------------------------------------------------
    if len(G) == 0:
        adj_tensor = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            size=(n_total, n_total)
        ).coalesce()
        return adj_tensor, df_sorted.index

    # --------------------------------------------------------------
    # 3. Nodes that appear in the graph
    # --------------------------------------------------------------
    present_nodes = sorted(G.nodes)
    n_sub = len(present_nodes)

    # --------------------------------------------------------------
    # 4. Build adjacency for present nodes only
    # --------------------------------------------------------------
    try:
        adj_sub = nx.to_scipy_sparse_array(G, nodelist=present_nodes,
                                          dtype=np.float32, format='csr')
    except AttributeError:
        adj_sub = nx.to_scipy_sparse_matrix(G, nodelist=present_nodes,
                                            dtype=np.float32, format='csr')

    # --------------------------------------------------------------
    # 5. Compute padding sizes
    # --------------------------------------------------------------
    pad_before = present_nodes[0]
    pad_after  = n_total - present_nodes[-1] - 1

    # --------------------------------------------------------------
    # 6. Create padding matrices (zero)
    # --------------------------------------------------------------
    row_pad_left  = csr_matrix((n_sub, pad_before), dtype=np.float32)
    row_pad_right = csr_matrix((n_sub, pad_after),  dtype=np.float32)
    col_pad       = csr_matrix((pad_after, n_total), dtype=np.float32)

    # --------------------------------------------------------------
    # 7. Stack horizontally: [pad_left | adj_sub | pad_right]
    # --------------------------------------------------------------
    adj_padded = hstack([row_pad_left, adj_sub, row_pad_right], format='csr')

    # --------------------------------------------------------------
    # 8. Stack vertically: add empty rows at bottom
    # --------------------------------------------------------------
    adj_padded = vstack([adj_padded, col_pad], format='csr')

    # --------------------------------------------------------------
    # 9. Convert to PyTorch sparse tensor
    # --------------------------------------------------------------
    coo = adj_padded.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
    values  = torch.from_numpy(coo.data).float()
    adj_tensor = torch.sparse_coo_tensor(indices, values,
                                         size=(n_total, n_total)).coalesce()

    return adj_tensor, df_sorted.index


def preprocess_data(raw_path: str, processed_path: str):
    print("Loading and preprocessing data...")
    df = pd.read_csv(raw_path)

    # Column name
    if 'Class' in df.columns:
        df.rename(columns={'Class': 'label'}, inplace=True)

    # Normalize numeric
    num_cols = [c for c in df.columns if c.startswith('V') or c in ['Amount', 'Time']]
    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / (df[num_cols].std() + 1e-8)

    # Velocity feature
    df = df.sort_values('Time').reset_index(drop=True)
    df['hour_bucket'] = (df['Time'] // 3600).astype(int)
    df['velocity_1h'] = df.groupby('hour_bucket')['Amount'].transform('count')
    df.drop(columns=['hour_bucket'], inplace=True)

    # Save
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

    # Build graph
    adj_tensor, _ = create_graph(df)

    feature_cols = [c for c in df.columns if c != 'label']
    features_tensor = torch.FloatTensor(df[feature_cols].values)
    labels_tensor   = torch.LongTensor(df['label'].values)

    graph_data = {
        'adj': adj_tensor,
        'features': features_tensor,
        'labels': labels_tensor
    }

    return df, graph_data
