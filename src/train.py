# File: src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.gnn import SimpleGNN
from src.models.random_forest import RandomForest
import numpy as np

def train_gnn(graph_data, epochs=50):
    model = SimpleGNN(graph_data['features'].shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    adj = torch.sparse.FloatTensor(
        torch.LongTensor([graph_data['adj'].row, graph_data['adj'].col]),
        torch.FloatTensor(graph_data['adj'].data)
    ).coalesce()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data['features'], adj)
        loss = criterion(out, graph_data['labels'])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"GNN Epoch {epoch} Loss: {loss.item():.4f}")
    return model

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
