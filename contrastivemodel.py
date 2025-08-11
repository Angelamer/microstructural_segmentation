import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap


torch.manual_seed(42)
np.random.seed(42)

class ProjectionHead(nn.Module):
    """Project Head for projecting multiple modes to the same space"""
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ContrastiveModel(nn.Module):
    def __init__(self, feature_dim, element_dim, latent_dim=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.element_dim = element_dim
        # Reduced features for kikuchi patterns (pca scores, cnmf weights, latent representation...)
        self.feature_encoder = ProjectionHead(feature_dim, latent_dim)
        # element (chemical) encoder
        self.element_encoder = ProjectionHead(element_dim, latent_dim)
        # Shared projection
        self.shared_projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, features, elements):
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, got {features.shape[1]}")
        if elements.shape[1] != self.element_dim:
            raise ValueError(f"Expected {self.element_dim} elements, got {elements.shape[1]}")
        feature_emb = self.feature_encoder(features)
        
        element_emb = self.element_encoder(elements)
        
        feature_proj = self.shared_projection(feature_emb)
        element_proj = self.shared_projection(element_emb)
        
        return feature_proj, element_proj
    
    

class NTXentLoss(nn.Module):
    """Normalized Temperature-Scaled Cross-Entropy Loss"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z_i, z_j):
        """
        z_i, z_j: Different modal representative from one sample (batch_size, latent_dim)
        """
        batch_size = z_i.size(0)
        
        # Cosine similarity
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, latent_dim)
        sim_matrix = self.cosine_sim(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        # Create labels
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)
        
        # Exclude the positive samples (diagonal)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)
        
        # Loss calculation
        positives = sim_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = sim_matrix[~labels.bool()].view(sim_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z.device)
        
        return self.criterion(logits, labels)
    
    
class FeatureElementDataset(Dataset):
        def __init__(self, data, feature_type):
            self.data = data
            self.feature_type = feature_type
            
            if feature_type == 'pca':
                self.feature_cols = [col for col in data.columns if 'PC_' in col]
            else:  # 'cnmf'
                self.feature_cols = [col for col in data.columns if 'cNMF_' in col]
                
            self.element_cols = [col for col in data.columns if col in ['O', 'Mg', 'Al', 'Si', 'Ti', 'Mn', 'Fe']]
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            features = torch.tensor(self.data.iloc[idx][self.feature_cols].values, dtype=torch.float32)
            elements = torch.tensor(self.data.iloc[idx][self.element_cols].values, dtype=torch.float32)
            return features, elements
        

def train_and_evaluate(feature_type, train_data, test_data, output_dim =2, num_epochs=100, batch_size=32, lr=1e-3):
    """
    Train and evaluate a comparative learning model for a specific feature type (PCA or cNMF)

    Args:
        feature_type: 'pca' or 'cnmf'
        train_data: training data DataFrame
        test_data: test data DataFrame
        num_epochs: number of training epochs
        batch_size: batch size
        lr: learning rate

    Returns:
        model: trained model
        train_losses: training loss history
        test_losses: test loss history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Create the dataset and loader
    train_dataset = FeatureElementDataset(train_data, feature_type)
    test_dataset = FeatureElementDataset(test_data, feature_type)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and loss
    feature_dim = len(train_dataset.feature_cols)
    element_dim = len(train_dataset.element_cols)
    model = ContrastiveModel(feature_dim, element_dim, output_dim).to(device)
    criterion = NTXentLoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    train_losses = []
    test_losses = []
    
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        for features, elements in train_loader:
            features = features.to(device)
            elements = elements.to(device)
            
            # Forward propagation
            feature_proj, element_proj = model(features, elements)
            
            # Loss calculation
            loss = criterion(feature_proj, element_proj)
            
            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Test
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for features, elements in test_loader:
                features = features.to(device)
                elements = elements.to(device)
                
                # Forward
                feature_proj, element_proj = model(features, elements)
                
                # Calculate the loss
                loss = criterion(feature_proj, element_proj)
                epoch_test_loss += loss.item()
        
        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], {feature_type.upper()} Model - "
            f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    
    return model, train_losses, test_losses

