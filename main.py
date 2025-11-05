# File: main.py
# Project: eco — header (replace with your text)
#!/usr/bin/env python3
"""
Main entry point: full pipeline
1. Preprocess data
2. Generate synthetic data
3. Train models
4. Evaluate & save results
"""
import os
from src.preprocess import preprocess_data
from src.generate_synthetic import generate_synthetic
from src.train import train_models
from src.evaluate import evaluate_and_save

def main():
    raw_path = 'data/raw/creditcard.csv'  # User must place ULB dataset here
    processed_path = 'data/processed/processed_creditcard.csv'
    synthetic_path = 'data/synthetic/wgan_synthetic.csv'

    # 1. Preprocess
    df, graph_data = preprocess_data(raw_path, processed_path)

    # 2. Generate synthetic (optional augmentation)
    generate_synthetic(df, synthetic_path, n_samples=50000)

    # 3. Train
    models = train_models(df, graph_data)

    # 4. Evaluate
    evaluate_and_save(models, df, graph_data)

    print("Pipeline completed! Check figures/ and tables/")

if __name__ == "__main__":
    main()
