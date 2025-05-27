import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader, TensorDataset
from .transformer_model import PatchAutoencoder, SimpleTransformer
from sklearn.decomposition import PCA
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix

def load_model(model_path: str, input_dim: int, latent_dim: int):
    model = PatchAutoencoder(latent_dim=latent_dim, input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_data(file_path, dataset_name):
    """Load and preprocess HSI data from .mat file."""
    try:
        data = sio.loadmat(file_path)
        key = {
            'pavia': 'paviaU',
            'salinas': 'salinas_corrected',
            'indian': 'indian_pines_corrected'
        }.get(dataset_name, list(data.keys())[-1])
        
        data = data[key].astype(np.float32)
        data = data / np.max(data)
        h, w, bands = data.shape
        data_reshaped = data.reshape(-1, bands)
        pca = PCA(n_components=30 if dataset_name != 'indian' else 40)
        return pca.fit_transform(data_reshaped).reshape(h, w, -1)
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {str(e)}")

def train_model(model, train_loader, epochs=10, lr=0.001):
    """Train the autoencoder."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output, _ = model(batch[0])
            loss = criterion(output, batch[0])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
    return model

def load_test_data(file_path, dataset_name):
    """Load test data and labels from .mat file."""
    try:
        data = sio.loadmat(file_path)
        data_key = {
            'pavia': 'paviaU',
            'salinas': 'salinas_corrected',
            'indian': 'indian_pines_corrected'
        }.get(dataset_name, list(data.keys())[-1])
        label_key = {
            'pavia': 'paviaU_gt',
            'salinas': 'salinas_gt',
            'indian': 'indian_pines_gt'
        }.get(dataset_name, None)

        if label_key is None or label_key not in data:
            raise ValueError("Ground truth labels not found in the dataset")

        hsi_data = data[data_key].astype(np.float32) / np.max(data[data_key])
        labels = data[label_key]

        h, w, bands = hsi_data.shape
        data_reshaped = hsi_data.reshape(-1, bands)
        labels_reshaped = labels.reshape(-1)

        pca = PCA(n_components=30 if dataset_name != 'indian' else 40)
        data_pca = pca.fit_transform(data_reshaped)

        return data_pca, labels_reshaped
    except Exception as e:
        raise ValueError(f"Loading test data failed: {str(e)}")

def evaluate_model(model, test_data, test_labels):
    """Evaluate model on test data and compute accuracy and confusion matrix."""
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(test_data, dtype=torch.float32)
        outputs = model.predict(input_tensor)
        predictions = outputs[0].argmax(dim=1).numpy() if outputs else None

    if predictions is None:
        raise ValueError("Model prediction failed or returned None")

    acc = accuracy_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)

    return acc, cm.tolist()

def evaluate_ae_model(model, test_data, test_labels, threshold=0.01):
    """
    Evaluate AETransformer model on test data by computing reconstruction error,
    applying threshold to classify anomalies, and calculating accuracy and confusion matrix.
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(test_data, dtype=torch.float32)
        x_recon, _ = model.predict(input_tensor)
        recon_error = torch.mean((input_tensor - x_recon) ** 2, dim=1).numpy()

    predictions = (recon_error > threshold).astype(int)

    acc = accuracy_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)

    return acc, cm.tolist()
