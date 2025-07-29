import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pandas as pd

from os.path import exists, join
import numpy as np

from .autoencoder import AutoencoderTrainer

def fit_transform_with_autoencoder(df: pd.DataFrame, 
                                   latent_space = 10,
                                    epochs=50, # Reduced epochs for quicker demonstration
                                    learning_rate=0.001,
                                    batch_size=64,
                                    train_split_ratio=0.8, 
                                    checkpoint_dir = "my_autoencoder_checkpoints", 
                                    replace = False, 
                                    *args, 
                                    **kwargs): 
    df = df.copy() 
    features = df.select_dtypes(include=np.number).columns.tolist()
    features_df = df[features]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    
    best_checkpoint_model_fullpath = join(checkpoint_dir, AutoencoderTrainer.best_checkpoint_naming)
    if exists(best_checkpoint_model_fullpath) and (replace==False) : 
        print(f"The saved model checkpoint is already here in {best_checkpoint_model_fullpath}")
        print(f"If you want to train new, please set replace = True")

        print("---Initializing the Trainer---")
        trainer = AutoencoderTrainer(
                input_dim=scaled_features.shape[-1],
                latent_dim=latent_space,
                epochs=epochs, # Reduced epochs for quicker demonstration
                lr=learning_rate,
                batch_size=batch_size,
                split_ratio=train_split_ratio,
                checkpoint_save_dir=checkpoint_dir, # Custom checkpoint directory 
                checkpoint_name=AutoencoderTrainer.best_checkpoint_naming, 
                **kwargs
                    )
    else: 
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

