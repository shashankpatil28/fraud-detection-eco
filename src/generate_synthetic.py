# File: src/generate_synthetic.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

def gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1).to(device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    d_inter = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_inter, inputs=interpolates,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True, retain_graph=True
    )[0]
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def generate_synthetic(df, save_path, n_samples=50000, epochs=1000):
    print("Generating synthetic fraud data using WGAN-GP...")
    fraud_df = df[df['label'] == 1].copy()
    if len(fraud_df) == 0:
        print("No fraud samples! Skipping.")
        return

    # --- CHANGES START ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating synthetic data on {device}...")
    # --- CHANGES END ---

    features = fraud_df.drop(columns=['label']).values
    # Normalize features for WGAN
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0) + 1e-8
    features = (features - feat_mean) / feat_std
    features = torch.FloatTensor(features).to(device)

    # --- CHANGES START ---
    G = Generator(100, features.shape[1]).to(device)
    C = Critic(features.shape[1]).to(device)
    # --- CHANGES END ---

    opt_g = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(C.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for epoch in range(epochs):
        # Train Critic
        for _ in range(5):
            # --- CHANGES START ---
            noise = torch.randn(features.size(0), 100).to(device)
            fake = G(noise)
            real = features # real is already on device
            d_real = C(real).mean()
            d_fake = C(fake).mean()
            gp = gradient_penalty(C, real, fake, device)
            # --- CHANGES END ---
            loss_c = d_fake - d_real + 10 * gp
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

        # Train Generator
        # --- CHANGES START ---
        noise = torch.randn(features.size(0), 100).to(device)
        # --- CHANGES END ---
        fake = G(noise)
        loss_g = -C(fake).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch} | C: {loss_c.item():.3f} | G: {loss_g.item():.3f}")

    # Generate
    with torch.no_grad():
        # --- CHANGES START ---
        noise = torch.randn(n_samples, 100).to(device)
        synth = G(noise).cpu().numpy()
        # --- CHANGES END ---
        
        # De-normalize
        synth = synth * feat_std + feat_mean
        
        synth_df = pd.DataFrame(synth, columns=fraud_df.drop(columns=['label']).columns)
        synth_df['label'] = 1
        synth_df.to_csv(save_path, index=False)
        print(f"Saved {n_samples} synthetic samples to {save_path}")