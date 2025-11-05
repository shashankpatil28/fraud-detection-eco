# File: src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.gnn import SimpleGNN
from src.models.random_forest import RandomForest
import numpy as np

def train_gnn(graph_data, epochs=50):
    # 1. --- CHANGES START ---
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training GNN on {device}...")

    # Move data to the correct device
    features = graph_data['features'].to(device)
    labels = graph_data['labels'].to(device)
    adj = graph_data['adj'].to(device) # This is already the sparse tensor we need

    # Initialize model on the device
    model = SimpleGNN(features.shape[1]).to(device)
    # --- CHANGES END ---

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    # 2. --- REMOVED BLOCK ---
    # The old, incorrect code tried to rebuild the tensor:
    # adj = torch.sparse.FloatTensor(
    #     torch.LongTensor([graph_data['adj'].row, graph_data['adj'].col]),
    #     torch.FloatTensor(graph_data['adj'].data)
    # ).coalesce()
    # This was wrong, as graph_data['adj'] is already a torch.sparse_coo_tensor
    # --- END REMOVED BLOCK ---

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # 3. --- CHANGES START ---
        # Pass the tensors that are on the device
        out = model(features, adj)
        loss = criterion(out, labels)
        # --- CHANGES END ---
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"GNN Epoch {epoch} Loss: {loss.item():.4f}")
    
    # 4. --- CHANGES START ---
    # Return model to CPU for consistent evaluation
    return model.to("cpu")
    # --- CHANGES END ---

def train_models(df, graph_data):
    print("Training models...")
    X = df.drop(columns=['label']).values
    y = df['label'].values

    # GNN
    gnn_model = train_gnn(graph_data)

    # Random Forest
    rf = RandomForest(n_trees=10, max_depth=6)
    rf.fit(X, y)

    return {'gnn': gnn_model, 'rf': rf}