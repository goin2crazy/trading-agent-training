import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import numpy as np

from .autoencoder import AutoencoderTrainer

def fit_transform_with_autoencoder(df: pd.DataFrame, 
                                   latent_space = 10,
                                    epochs=50, # Reduced epochs for quicker demonstration
                                    learning_rate=0.001,
                                    batch_size=64,
                                    train_split_ratio=0.8, 
                                    checkpoint_dir = "my_autoencoder_checkpoints", 
                                    *args, 
                                    **kwargs): 
    df = df.copy() 
    features = df.select_dtypes(include=np.number).columns.tolist()
    features_df = df[features]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)

    print("---Initializing the Trainer---")
    trainer = AutoencoderTrainer(
            input_dim=scaled_features.shape[-1],
            latent_dim=latent_space,
            epochs=epochs, # Reduced epochs for quicker demonstration
            lr=learning_rate,
            batch_size=batch_size,
            split_ratio=train_split_ratio,
            checkpoint_save_dir=checkpoint_dir, # Custom checkpoint directory
            **kwargs
                )
    print("---Training starts---")
    trainer.train(scaled_features)
    
    # Use the entire DataFrame for encoding
    print(f"Encoding entire DataFrame of shape: {scaled_features.shape}")
    encoded_latent_representation = trainer.encode_df(df)
    print("\nEncoded Latent Representation (first 5 samples):")
    print(encoded_latent_representation.head())
    print(f"Shape of latent representation: {encoded_latent_representation.shape}")
    return encoded_latent_representation, trainer

