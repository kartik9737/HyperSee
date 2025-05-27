import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import scipy.io as sio
from collections import Counter

class PatchAutoencoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(PatchAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def load_dataset(name, hsi_path, gt_path):
    data = sio.loadmat(hsi_path)[list(sio.loadmat(hsi_path).keys())[-1]].astype(np.float32)
    gt = sio.loadmat(gt_path)[list(sio.loadmat(gt_path).keys())[-1]]
    return data, gt

def preprocess(data, gt, dataset_name):
    h, w, bands = data.shape
    if dataset_name == 'indian':
        noisy_bands = [b for b in (list(range(104, 109)) + list(range(150, 164)) + [220]) if b < bands]
        data = np.delete(data, noisy_bands, axis=2)
    scaler = MinMaxScaler()
    data_reshaped = data.reshape(-1, data.shape[2])
    data_scaled = scaler.fit_transform(data_reshaped)
    pca_components = 30 if dataset_name == 'pavia' else 40
    pca = PCA(n_components=pca_components)
    data_pca = pca.fit_transform(data_scaled)
    data_pca = data_pca.reshape(h, w, -1)
    data_pca = (data_pca - np.min(data_pca)) / (np.max(data_pca) - np.min(data_pca))
    return data_pca, gt, h, w, pca_components

def extract_patches(data, gt, patch_size):
    h, w, _ = data.shape
    margin = patch_size // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
    padded_gt = np.pad(gt, ((margin, margin), (margin, margin)), mode='reflect')
    patches, labels, coords = [], [], []
    for i in range(margin, margin + h):
        for j in range(margin, margin + w):
            patch = padded_data[i - margin:i + margin, j - margin:j + margin, :]
            label = padded_gt[i, j]
            if label != 0:
                patches.append(patch)
                labels.append(label)
                coords.append((i - margin, j - margin))
    return np.array(patches), np.array(labels), np.array(coords), h, w

def get_pca_rgb_image(data, n_components=3):
    h, w, d = data.shape
    reshaped_data = data.reshape(-1, d)
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(reshaped_data)
    transformed -= transformed.min(0)
    transformed /= transformed.max(0)
    rgb = (transformed * 255).astype(np.uint8).reshape(h, w, 3)
    return rgb

def overlay_anomalies_on_rgb(
    rgb_img, coords, y_test, anomalies, label_to_name, descriptions=None,
    title="Anomaly Classification Map (on RGB)", save_path=None
):
    h, w, _ = rgb_img.shape
    valid = [i for i, (x, y) in enumerate(coords) if 0 <= x < h and 0 <= y < w]
    coords = np.array(coords)[valid]
    y_test = np.array(y_test)[valid]
    anomalies = np.array(anomalies)[valid]
    if len(anomalies) != len(coords):
        raise ValueError(f"Length mismatch: anomalies {len(anomalies)} vs coords {len(coords)}")
    anomaly_coords = coords[anomalies]
    anomaly_labels = y_test[anomalies]
    unique_labels = np.unique(anomaly_labels)
    palette = sns.color_palette("tab20", len(unique_labels))
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    plt.figure(figsize=(14, 10))
    plt.imshow(rgb_img)
    plt.axis('off')
    for (x, y), label in zip(anomaly_coords, anomaly_labels):
        plt.scatter(y, x, s=30, color=color_map[label], edgecolors='black', linewidth=0.5)
    patches = []
    for label in unique_labels:
        name = label_to_name.get(label, f"Class {label}")
        desc = descriptions.get(name, "") if descriptions else ""
        label_text = f"{name}: {desc}" if desc else name
        patches.append(mpatches.Patch(color=color_map[label], label=label_text))
    plt.legend(
        handles=patches, loc='upper right', bbox_to_anchor=(1, 1),
        borderaxespad=0., fontsize=8, title="Legend", frameon=True
    )
    plt.title("📍 " + title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_classification_pipeline(hsi_filename, gt_filename, dataset_name, uploads_dir):
    dataset_metadata = {
        'pavia': {
            'label_to_name': {
                1: 'Asphalt',
                2: 'Meadows',
                3: 'Gravel',
                4: 'Trees',
                5: 'Painted metal sheets',
                6: 'Bare Soil',
                7: 'Bitumen',
                8: 'Self-Blocking Bricks',
                9: 'Shadows'
            },
            'descriptions': {
                'Asphalt': "Urban road surface.",
                'Meadows': "Large grassy areas.",
                'Gravel': "Loose rock surface.",
                'Trees': "Vegetation cover.",
                'Painted metal sheets': "Reflective urban structures.",
                'Bare Soil': "Dry, unplanted earth.",
                'Bitumen': "Waterproof construction surface.",
                'Self-Blocking Bricks': "Interlocking brick patterns.",
                'Shadows': "Shadowed zones in urban area."
            }
        },
        'indian': {
            'label_to_name': {
                1: 'Alfalfa',
                2: 'Corn-notill',
                3: 'Corn-mintill',
                4: 'Corn',
                5: 'Grass-pasture',
                6: 'Grass-trees',
                7: 'Grass-pasture-mowed',
                8: 'Hay-windrowed',
                9: 'Oats',
                10: 'Soybean-notill',
                11: 'Soybean-mintill',
                12: 'Soybean-clean',
                13: 'Wheat',
                14: 'Woods',
                15: 'Buildings-Grass-Trees-Drives',
                16: 'Stone-Steel-Towers'
            },
            'descriptions': {
                'Alfalfa': "Perennial flowering plant.",
                'Corn-notill': "Corn grown without tillage.",
                'Corn-mintill': "Corn with minimal tillage.",
                'Corn': "Fully cultivated corn.",
                'Grass-pasture': "Grazing grassland.",
                'Grass-trees': "Mixed vegetation.",
                'Grass-pasture-mowed': "Cut pasture field.",
                'Hay-windrowed': "Dry hay laid in rows.",
                'Oats': "Cultivated oats.",
                'Soybean-notill': "Soybean without tillage.",
                'Soybean-mintill': "Soybean with light tillage.",
                'Soybean-clean': "Soybean grown cleanly.",
                'Wheat': "Wheat crop.",
                'Woods': "Dense trees.",
                'Buildings-Grass-Trees-Drives': "Urban mix with greenery.",
                'Stone-Steel-Towers': "Tall infrastructure features."
            }
        },
        'salinas': {
            'label_to_name': {
                1: "Broccoli_green_weeds_1",
                2: "Broccoli_green_weeds_2",
                3: "Fallow",
                4: "Fallow_rough_plow",
                5: "Fallow_smooth",
                6: "Stubble",
                7: "Celery",
                8: "Grapes_untrained",
                9: "Soil_vinyard_develop",
                10: "Corn_senesced_green_weeds",
                11: "Lettuce_romaine_4wk",
                12: "Lettuce_romaine_5wk",
                13: "Lettuce_romaine_6wk",
                14: "Lettuce_romaine_7wk",
                15: "Vinyard_untrained",
                16: "Vinyard_vertical_trellis"
            },
            'descriptions': {
                "Broccoli_green_weeds_1": "Weeds in broccoli field type 1.",
                "Broccoli_green_weeds_2": "Weeds in broccoli field type 2.",
                "Fallow": "Unplanted agricultural field.",
                "Fallow_rough_plow": "Rough-plowed, uncultivated land.",
                "Fallow_smooth": "Smooth, barren agricultural field.",
                "Stubble": "Remnants of harvested crops.",
                "Celery": "Celery vegetation detected.",
                "Grapes_untrained": "Grapevines not trained on trellis.",
                "Soil_vinyard_develop": "Soil under vineyard development.",
                "Corn_senesced_green_weeds": "Dried corn with weeds.",
                "Lettuce_romaine_4wk": "4-week romaine lettuce.",
                "Lettuce_romaine_5wk": "5-week romaine lettuce.",
                "Lettuce_romaine_6wk": "6-week romaine lettuce.",
                "Lettuce_romaine_7wk": "7-week romaine lettuce.",
                "Vinyard_untrained": "Untrained vineyard vines.",
                "Vinyard_vertical_trellis": "Vineyard with vertical trellis."
            }
        }
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hsi_path = os.path.join(uploads_dir, hsi_filename)
    gt_path = os.path.join(uploads_dir, gt_filename)
    data, gt = load_dataset(dataset_name, hsi_path, gt_path)
    data_pca, gt, h, w, pca_components = preprocess(data, gt, dataset_name)
    patch_size = 5 if dataset_name == 'indian' or dataset_name == 'salinas' else 3
    patches, labels, coords, h, w = extract_patches(data_pca, gt, patch_size)
    patches = patches.reshape(patches.shape[0], -1)
    input_dim = patches.shape[1]
    latent_dim = 32
    patches_tensor = torch.tensor(patches, dtype=torch.float32).to(device)
    dataset = TensorDataset(patches_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = PatchAutoencoder(latent_dim=latent_dim, input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        patches, labels, coords, test_size=0.2, stratify=labels, random_state=42
    )
    patches_tensor_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    patches_tensor_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    patches_tensor_train = patches_tensor_train.view(patches_tensor_train.size(0), -1)
    patches_tensor_test = patches_tensor_test.view(patches_tensor_test.size(0), -1)
    input_dim = patches_tensor_train.size(1)
    latent_dim = 32
    model = PatchAutoencoder(latent_dim=latent_dim, input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    num_epochs = 20
    model.train()
    dataset_train = TensorDataset(patches_tensor_train)
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader_train:
            x = batch[0]
            optimizer.zero_grad()
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    model.eval()
    with torch.no_grad():
        latent_features_train = model.encoder(patches_tensor_train).cpu().numpy()
        latent_features_test = model.encoder(patches_tensor_test).cpu().numpy()
        reconstructions = model(patches_tensor_test).detach().cpu().numpy()
    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(latent_features_train, y_train)
    y_pred = svm.predict(latent_features_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    confusion_matrix_path = os.path.join(uploads_dir, f'confusion_matrix_{dataset_name}.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    label_names = dataset_metadata.get(dataset_name, {}).get('label_to_name', {})
    descriptions = dataset_metadata.get(dataset_name, {}).get('descriptions', {})
    rgb_img = get_pca_rgb_image(data)
    reconstruction_errors = np.mean((X_test.reshape(X_test.shape[0], -1) - reconstructions) ** 2, axis=1)
    threshold = np.percentile(reconstruction_errors, 95)
    anomalies = reconstruction_errors > threshold
    save_path = os.path.join(uploads_dir, f'anomaly_overlay_{dataset_name}.png')
    overlay_anomalies_on_rgb(
        rgb_img, coords_test, y_test, anomalies,
        label_names, descriptions,
        save_path=save_path
    )
    # Save classification report as CSV
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
    classification_report_df = pd.DataFrame(classification_report_dict).transpose()
    csv_path = os.path.join(uploads_dir, f'classification_report_{dataset_name}.csv')
    classification_report_df.to_csv(csv_path)
    return {
        'confusion_matrix_path': confusion_matrix_path,
        'anomaly_overlay_path': save_path,
        'classification_report_path': csv_path,
        'classification_report': classification_report_dict
    }
