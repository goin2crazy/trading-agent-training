import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
import numpy as np

from .model import Autoencoder_Deep, Autoencoder_Simple

# --- 1. Custom Dataset ---
class DataFrameDataset(Dataset):
    def __init__(self, data_array):
        self.data = torch.tensor(data_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --- 3. Autoencoder Trainer ---
class AutoencoderTrainer:
    def __init__(self, 
                 input_dim, 
                 latent_dim, 
                 lr=1e-3, 
                 epochs=100, 
                 batch_size=32, 
                 split_ratio=0.8, 
                 checkpoint_save_dir = "autoencoder_checkpoints", 
                 checkpoint_name = None, 
                 deep=True, 
                 tanh = True):
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        # Defining what type of architecture going to be used
        self.deep_arch = deep
        self.tanh = tanh

        if self.deep_arch: 
            self.model = Autoencoder_Deep(input_dim, latent_dim, tanh=self.tanh)
        else: 
            self.model = Autoencoder_Simple(input_dim, latent_dim)

        self.checkpoint_dir=checkpoint_save_dir
        self.checkpoint_name = checkpoint_name
        self.checkpointed_accuracies = [] 

        # Ensure the data save directory exists
        if not os.path.exists(checkpoint_save_dir):
            os.makedirs(checkpoint_save_dir)

        if type(checkpoint_name) == str: 
            try: 
                self.load_checkpoint(os.path.join(self.checkpoint_dir, self.checkpoint_name))
            except: 
                print("Chackpoint did not find, initializing new model")

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.best_val_accuracy = 1

    def _calculate_r2(self, original, reconstructed):
        ss_res = torch.sum((original - reconstructed) ** 2)
        ss_tot = torch.sum((original - torch.mean(original)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2.item() * 100

    def train(self, dataframe: pd.DataFrame):
        full_dataset = DataFrameDataset(dataframe)
        train_size = int(self.split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for data in train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, data)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_r2 = 0.0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(self.device)
                    output = self.model(data)
                    val_loss += self.criterion(output, data).item()
                    val_r2 += self._calculate_r2(data, output)

            avg_val_loss = val_loss / len(val_loader)
            avg_val_r2 = val_r2 / len(val_loader)

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, R2: {avg_val_r2:.2f}%")
            if avg_val_r2 > self.best_val_accuracy:
                self.best_val_accuracy = avg_val_r2

                checkpoint_filename = os.path.join(self.checkpoint_dir, f"autoencoder_best.pth")
                self.save_checkpoint(checkpoint_filename, avg_val_r2, epoch)
                print(f"--- Checkpoint saved at {avg_val_r2:.2f}% validation accuracy! ---")
        
        checkpoint_filename = os.path.join(self.checkpoint_dir, f"autoencoder_final.pth")
        self.save_checkpoint(checkpoint_filename, avg_val_r2, epoch)


    def save_checkpoint(self, filepath: str, accuracy: float, epoch: int):
        """
        Saves the current model state and training progress.

        Args:
            filepath (str): The path where the checkpoint file will be saved.
            accuracy (float): The validation accuracy at the time of saving.
            epoch (int): The current epoch number.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': accuracy,
            'latent_dim': self.latent_dim,
            'input_dim': self.input_dim,
            'checkpointed_accuracies': list(self.checkpointed_accuracies) # Save as list for serialization
        }
        torch.save(checkpoint, filepath)
        print(f"Model checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """
        Loads a model from a saved checkpoint.

        Args:
            filepath (str): The path to the checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Re-initialize model if dimensions don't match (or assert for strictness)
        if checkpoint['input_dim'] != self.input_dim or checkpoint['latent_dim'] != self.latent_dim:
            print(f"Warning: Checkpoint dimensions ({checkpoint['input_dim']}, {checkpoint['latent_dim']}) "
                  f"do not match current model dimensions ({self.input_dim}, {self.latent_dim}). "
                  f"Re-initializing model with checkpoint dimensions.")
            
            if self.deep_arch: 
                self.model =Autoencoder_Deep(checkpoint['input_dim'], checkpoint['latent_dim'])
            else: 
                self.model =Autoencoder_Simple(checkpoint['input_dim'], checkpoint['latent_dim'])

            self.model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) # Re-initialize optimizer too

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_accuracy = checkpoint.get('val_accuracy', -1.0) # Use .get for backward compatibility
        self.checkpointed_accuracies = set(checkpoint.get('checkpointed_accuracies', []))
        print(f"Model loaded from {filepath} (Epoch: {checkpoint['epoch']}, "
              f"Val Accuracy: {self.best_val_accuracy:.2f}%)")
        self.model.eval() # Set to eval mode after loading

    # --- Simple Inference Interface ---
    def encode(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Encodes a pandas DataFrame into its latent representation.
        Automatically handles large datasets by splitting into batches.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            torch.Tensor: The concatenated latent space representation for the entire DataFrame.
        """
        self.model.eval() # Set model to evaluation mode
        encoded_representations = []
        
        # Create a DataLoader for the input DataFrame
        inference_dataset = DataFrameDataset(data)
        # Use the same batch_size as training for consistency, or adjust if needed for inference
        inference_loader = DataLoader(inference_dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad(): # Disable gradient calculations for inference
            for batch_data in inference_loader:
                batch_data = batch_data.to(self.device) # Move batch to the device
                encoded_batch = self.model.encoder(batch_data) # Encode the batch
                encoded_representations.append(encoded_batch.cpu()) # Move to CPU and collect

        # Concatenate all encoded batches into a single tensor
        return torch.cat(encoded_representations, dim=0)
    
    def encode_df(self, df: pd.DataFrame) ->pd.DataFrame:
        # Make a copy to avoid modifying the original DataFrame passed in
        df = df.copy() 

        df_copy_ = df.copy() 

        features = df.select_dtypes(include=np.number).columns.tolist()
        df_copy = df[features]
        # Encode the DataFrame, which handles batching internally
        encoded_tensors = self.encode(df_copy.values).numpy()
        print(f"Shape of encoded tensors: {encoded_tensors.shape}")

        # Create new column names for the encoded features
        encoded_col_names = [f'enc_{i}' for i in range(encoded_tensors.shape[1])]

        # Create a new DataFrame from the encoded tensors, using the original DataFrame's index
        # This ensures proper alignment when concatenating
        encoded_df = pd.DataFrame(encoded_tensors, columns=encoded_col_names)

        # Concatenate the original DataFrame (copy) with the new encoded features DataFrame
        # axis=1 ensures columns are added
        df_final = pd.concat([df_copy_, encoded_df], axis=1)

        return df_final

    def decode(self, latent_representation: torch.Tensor) -> pd.DataFrame:
        """
        Decodes a latent representation back into the original data space.

        Args:
            latent_representation (torch.Tensor): The latent space tensor.

        Returns:
            pd.DataFrame: The reconstructed data as a pandas DataFrame.
        """
        self.model.eval() # Set model to evaluation mode
        reconstructed_data_batches = []

        # Create a DataLoader for the latent representation
        # This allows decoding large latent tensors in batches
        latent_dataset = torch.utils.data.TensorDataset(latent_representation)
        latent_loader = DataLoader(latent_dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad(): # Disable gradient calculations for inference
            for batch_z in latent_loader:
                # batch_z from DataLoader will be a list/tuple of tensors, we need the first element
                batch_z = batch_z[0].to(self.device)
                decoded_batch = self.model.decoder(batch_z)
                reconstructed_data_batches.append(decoded_batch.cpu())

        # Concatenate all decoded batches into a single tensor and convert to DataFrame
        reconstructed_tensor = torch.cat(reconstructed_data_batches, dim=0)
        return pd.DataFrame(reconstructed_tensor.numpy())

