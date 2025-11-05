# File: src/evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from src.utils import save_fig, save_table, accuracy, precision_recall_f1


def evaluate_and_save(models, df, graph_data):
    """
    Evaluate GNN, RandomForest and an ensemble.
    All metrics are computed with pure numpy.
    """
    X = df.drop(columns=['label']).values
    y = df['label'].values.astype(int)

    # ---------- GNN inference ----------
    gnn = models['gnn']
    gnn.eval() # Model is on CPU from train.py
    with torch.no_grad():
        # 1. --- CHANGES START ---
        # Get tensors from graph_data (they are on CPU)
        features = graph_data['features']
        adj_sp = graph_data['adj']
        
        # --- REMOVED BLOCK ---
        # The old, incorrect code:
        # adj_coo = graph_data['adj'].tocoo()
        # indices = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)))
        # values  = torch.FloatTensor(adj_coo.data)
        # adj_sp  = torch.sparse.FloatTensor(indices, values,
        #                                    torch.Size(adj_coo.shape)).coalesce()
        # --- END REMOVED BLOCK ---

        out = gnn(features, adj_sp)
        gnn_pred = out.argmax(dim=1).cpu().numpy()
        # --- CHANGES END ---

    # ---------- RandomForest ----------
    rf_pred_proba = models['rf'].predict_proba(X)
    rf_pred = (rf_pred_proba > 0.5).astype(int)

    # ---------- Ensemble (simple average voting) ----------
    # Note: Your context mentioned Isolation Forest, but the pipeline
    # (main.py, train.py, evaluate.py) only implements GNN and RF.
    # I am keeping your existing logic intact.
    ensemble_pred = ((gnn_pred + rf_pred) >= 1).astype(int)

    # ---------- Compute metrics ----------
    results = []
    for name, pred in [('GNN', gnn_pred),
                       ('RF',  rf_pred),
                       ('Ensemble', ensemble_pred)]:
        acc = accuracy(y, pred)
        p, r, f1 = precision_recall_f1(y, pred, pos_label=1)
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': p,
            'Recall': r,
            'F1': f1
        })

    metrics_df = pd.DataFrame(results)
    save_table(metrics_df, 'performance_metrics')

    # ---------- Accuracy bar plot ----------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(metrics_df['Model'], metrics_df['Accuracy'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    # Adjust ylim for visibility, 0.997 was too high for some datasets
    ax.set_ylim(0.98, 1.0) 
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy')
    save_fig(fig, 'accuracy_bar')

    # ---------- Simulated growth-metric improvement ----------
    sim_df = pd.DataFrame({
        'Scenario': ['Before Detection', 'After Detection'],
        'CAC': [120, 96],          # 20% reduction
        'LTV': [300, 372],         # 24% increase
        'CAC_to_LTV_Ratio': [120/300, 96/372]
    })
    save_table(sim_df, 'growth_impact')

    # ---------- Bar chart for CAC / LTV ----------
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(sim_df['Scenario'], sim_df['CAC'], label='CAC', alpha=0.7)
    ax2.bar(sim_df['Scenario'], sim_df['LTV'], bottom=sim_df['CAC'],
            label='LTV', alpha=0.5)
    ax2.set_ylabel('Value ($)')
    ax2.set_title('CAC vs LTV Before / After Fraud Detection')
    ax2.legend()
    save_fig(fig2, 'cac_ltv')

    print("Evaluation complete – all tables & figures saved!")