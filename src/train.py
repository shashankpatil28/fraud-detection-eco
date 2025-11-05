# File: src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.gnn import SimpleGNN
from src.models.random_forest import RandomForest
from src.models.isolation_forest import IsolationForest # <-- NEW IMPORT
import numpy as np

def train_gnn(graph_data, epochs=50):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training GNN on {device}...")

    # Move data to the correct device
    features = graph_data['features'].to(device)
    labels = graph_data['labels'].to(device)
    adj = graph_data['adj'].to(device) # This is already the sparse tensor we need

    # Initialize model on the device
    model = SimpleGNN(features.shape[1]).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(features, adj)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"GNN Epoch {epoch} Loss: {loss.item():.4f}")
    
    # Return model to CPU for consistent evaluation
    return model.to("cpu")


def train_models(df, graph_data):
    print("Training models...")
    X = df.drop(columns=['label']).values
    y = df['label'].values

    # GNN
    gnn_model = train_gnn(graph_data)

    # Random Forest
    rf = RandomForest(n_trees=10, max_depth=6)
    rf.fit(X, y)

    # -----------------------------------------------------------------
    # --- NEW MODEL START ---
    # -----------------------------------------------------------------
    # Isolation Forest
    # Use sample_size=256 (standard) and n_trees=100 (standard)
    if_model = IsolationForest(n_trees=100, sample_size=256)
    # Fit on non-fraud data for anomaly detection (optional, but good practice)
    # For simplicity, we can fit on all data as in your context
    if_model.fit(X)
    # -----------------------------------------------------------------
    # --- NEW MODEL END ---
    # -----------------------------------------------------------------

    return {'gnn': gnn_model, 'rf': rf, 'if': if_model} # <-- ADDED IF