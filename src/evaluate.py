# File: src/evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
# --- IMPORT CHANGES START ---
from src.utils import save_fig, save_table, accuracy, precision_recall_f1, plot_confusion_matrix
# --- IMPORT CHANGES END ---


def evaluate_and_save(models, df, graph_data):
    """
    Evaluate GNN, RandomForest, IsolationForest and an ensemble.
    All metrics are computed with pure numpy.
    """
    X = df.drop(columns=['label']).values
    y = df['label'].values.astype(int)

    # ---------- GNN inference ----------
    gnn = models['gnn']
    gnn.eval() # Model is on CPU from train.py
    with torch.no_grad():
        features = graph_data['features']
        adj_sp = graph_data['adj']
        out = gnn(features, adj_sp)
        gnn_pred = out.argmax(dim=1).cpu().numpy()

    # ---------- RandomForest ----------
    rf_pred_proba = models['rf'].predict_proba(X)
    rf_pred = (rf_pred_proba > 0.5).astype(int)

    # -----------------------------------------------------------------
    # --- NEW MODEL INFERENCE START ---
    # -----------------------------------------------------------------
    if_model = models['if']
    if_scores = if_model.anomaly_score(X)
    
    # We need a threshold. Let's set the threshold to flag the same
    # percentage of transactions as the true fraud rate.
    fraud_rate = np.mean(y == 1)
    if_threshold = np.quantile(if_scores, 1 - fraud_rate)
    if_pred = (if_scores > if_threshold).astype(int)
    # -----------------------------------------------------------------
    # --- NEW MODEL INFERENCE END ---
    # -----------------------------------------------------------------


    # ---------- Ensemble (2-out-of-3 average voting) ----------
    # We sum the predictions and if 2 or more models vote "fraud", we call it fraud.
    ensemble_pred = ((gnn_pred + rf_pred + if_pred) >= 2).astype(int)

    # ---------- Compute metrics ----------
    results = []
    cm_dict = {} # To store confusion matrix data
    
    # --- EVALUATION LOOP CHANGES START ---
    for name, pred in [('GNN', gnn_pred),
                       ('RF',  rf_pred),
                       ('Isolation Forest', if_pred), # <-- ADDED IF
                       ('Ensemble', ensemble_pred)]:  # <-- ENSEMBLE IS NOW 3-WAY
        
        acc = accuracy(y, pred)
        # Get p, r, f1 AND confusion matrix (cm)
        p, r, f1, cm = precision_recall_f1(y, pred, pos_label=1) 
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': p,
            'Recall': r,
            'F1': f1
        })
        cm_dict[name] = cm # Save the cm tuple
    # --- EVALUATION LOOP CHANGES END ---

    metrics_df = pd.DataFrame(results)
    save_table(metrics_df, 'performance_metrics')

    # ---------- Plot 1: Accuracy bar plot (Your existing plot) ----------
    fig_acc, ax_acc = plt.subplots(figsize=(7, 5)) # Made wider
    models_list = metrics_df['Model']
    ax_acc.bar(models_list, metrics_df['Accuracy'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax_acc.set_ylim(0.98, 1.0) 
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Model Accuracy Comparison')
    save_fig(fig_acc, '1_accuracy_bar')

    # -----------------------------------------------------------------
    # --- NEW PLOT 2: Precision-Recall-F1 Bar Plot ---
    # -----------------------------------------------------------------
    fig_prf, ax_prf = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models_list))
    width = 0.25
    
    rects1 = ax_prf.bar(x - width, metrics_df['Precision'], width, label='Precision', color='#1f77b4')
    rects2 = ax_prf.bar(x,         metrics_df['Recall'],    width, label='Recall',    color='#ff7f0e')
    rects3 = ax_prf.bar(x + width, metrics_df['F1'],        width, label='F1-Score',  color='#2ca02c')

    ax_prf.set_ylabel('Score')
    ax_prf.set_title('Model Precision, Recall, and F1-Score')
    ax_prf.set_xticks(x, models_list)
    ax_prf.legend()
    ax_prf.set_ylim(0, 1.1) # Show full 0-1 range
    save_fig(fig_prf, '2_precision_recall_f1_bar')
    
    # -----------------------------------------------------------------
    # --- NEW PLOT 3: Confusion Matrix for Best Model ---
    # -----------------------------------------------------------------
    # We plot the CM for the Ensemble model, which is usually best
    ensemble_cm = cm_dict['Ensemble'] # (tp, fp, fn, tn)
    plot_confusion_matrix(ensemble_cm, 
                          classes=['Not Fraud (0)', 'Fraud (1)'], 
                          title='Ensemble Confusion Matrix')
    
    # -----------------------------------------------------------------
    # --- UPDATED PLOT 4: DATA-DRIVEN Economic Impact ---
    # -----------------------------------------------------------------
    
    # Get the Ensemble's Recall score (its ability to CATCH fraud)
    # This is the most important metric for this simulation
    ensemble_recall = metrics_df[metrics_df['Model'] == 'Ensemble']['Recall'].values[0]

    # Use the economic numbers from your project abstract/results:
    # - 23% LTV distortion from fraud
    # - 18% CAC reduction from detection
    
    base_ltv = 300
    base_cac = 120
    
    # Calculate the *actual* improvement based on your model's recall
    # If recall is 1.0 (100%), you prevent all 23% of distortion.
    # If recall is 0.5 (50%), you prevent 0.5 * 23% = 11.5% of distortion.
    ltv_distortion_prevented = 0.23 * ensemble_recall
    cac_waste_prevented = 0.18 * ensemble_recall
    
    # Calculate the new values
    new_ltv = base_ltv * (1 + ltv_distortion_prevented)
    new_cac = base_cac * (1 - cac_waste_prevented)

    # Create the DataFrame
    sim_df = pd.DataFrame({
        'Scenario': ['Before Detection', 'After Detection'],
        'CAC': [base_cac, new_cac],
        'LTV': [base_ltv, new_ltv],
        'CAC_to_LTV_Ratio': [base_cac / base_ltv, new_cac / new_ltv]
    })
    save_table(sim_df, 'growth_impact')

    # Plot the Bar chart (same as before, but with your new data)
    fig_eco, ax_eco = plt.subplots(figsize=(6, 4))
    ax_eco.bar(sim_df['Scenario'], sim_df['CAC'], label='CAC', alpha=0.7, color='#d62728') # Changed color
    ax_eco.bar(sim_df['Scenario'], sim_df['LTV'], bottom=sim_df['CAC'],
            label='LTV', alpha=0.5, color='#2ca02c') # Changed color
    ax_eco.set_ylabel('Value ($)')
    ax_eco.set_title(f'Economic Impact (Driven by {ensemble_recall*100:.1f}% Fraud Recall)')
    ax_eco.legend()
    save_fig(fig_eco, '4_cac_ltv_impact')

    print("Evaluation complete – all 4 plots & 2 tables saved!")