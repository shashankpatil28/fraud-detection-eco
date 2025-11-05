# File: src/preprocess.py
import pandas as pd
import numpy as np
import networkx as nx
import torch
from scipy.sparse import csr_matrix, hstack, vstack  # <-- Imports are fine


def create_graph(df: pd.DataFrame):
    """
    Build a transaction graph with safe padding for isolated nodes.
    Returns:
        adj_tensor : torch.sparse_coo_tensor (N, N) coalesced
        node_order : original index order (not used)
    """
    G = nx.Graph()
    df_sorted = df.sort_values('Time').reset_index(drop=True)
    n_total = len(df)

    # --------------------------------------------------------------
    # 1. --- CHANGE START ---
    # Add ALL nodes to the graph first.
    # This ensures that even isolated nodes (with no edges)
    # are in G, which fixes the NetworkXError.
    G.add_nodes_from(range(n_total))
    # --- CHANGE END ---
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # 2. Add edges (time < 1h and amount diff < $50)
    # --------------------------------------------------------------
    edges = []
    for i in range(len(df_sorted) - 1):
        r1 = df_sorted.iloc[i]
        r2 = df_sorted.iloc[i + 1]
        if (r2['Time'] - r1['Time']) < 3600 and abs(r2['Amount'] - r1['Amount']) < 50:
            edges.append((i, i + 1))
    G.add_edges_from(edges)


    # --------------------------------------------------------------
    # 3. If no edges → return zero matrix
    #    (We still check len(G.edges()) because G will have nodes)
    # --------------------------------------------------------------
    if len(G.edges()) == 0:
        adj_tensor = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            size=(n_total, n_total)
        ).cooalesce()
        return adj_tensor, df_sorted.index

    # --------------------------------------------------------------
    # 4. Build the full (N, N) adjacency matrix directly.
    #    This will now work because all nodes in the nodelist
    #    are guaranteed to be in G.
    # --------------------------------------------------------------
    try:
        adj_matrix = nx.to_scipy_sparse_array(G, nodelist=range(n_total),
                                              dtype=np.float32, format='csr')
    except AttributeError:
        # Fallback for older networkx versions
        adj_matrix = nx.to_scipy_sparse_matrix(G, nodelist=range(n_total),
                                                dtype=np.float32, format='csr')

    # --------------------------------------------------------------
    # 5. Convert to PyTorch sparse tensor
    # --------------------------------------------------------------
    coo = adj_matrix.tocoo()
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