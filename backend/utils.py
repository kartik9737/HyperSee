import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight

def preprocess(data, gt, dataset_name):
    h, w, bands = data.shape
    if dataset_name == 'indian':
        noisy_bands = [b for b in (list(range(104, 109)) + list(range(150, 164)) + [220]) if b < data.shape[-1]]
        data = np.delete(data, noisy_bands, axis=2)
    scaler = MinMaxScaler()
    data_reshaped = data.reshape(-1, data.shape[2])
    data_scaled = scaler.fit_transform(data_reshaped).astype(np.float32)  # convert to float32 to reduce memory usage
    pca_components = 30 if dataset_name != 'indian' else 40
    pca = PCA(n_components=pca_components)
    data_pca = pca.fit_transform(data_scaled)
    data_pca = data_pca.reshape(h, w, -1)
    return data_pca, gt, h, w, pca_components

def extract_patches(data, gt, patch_size):
    h, w, c = data.shape
    margin = patch_size // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
    padded_gt = np.pad(gt, ((margin, margin), (margin, margin)), mode='reflect')

    shape = (h, w, patch_size, patch_size, c)
    strides = (padded_data.strides[0], padded_data.strides[1], padded_data.strides[0], padded_data.strides[1], padded_data.strides[2])
    patches = np.lib.stride_tricks.as_strided(padded_data, shape=shape, strides=strides)
    patches = patches.reshape(-1, patch_size, patch_size, c)

    gt_flat = padded_gt[margin:margin+h, margin:margin+w].flatten()

    idx = np.where(gt_flat != 0)[0]
    patches = patches[idx]
    labels = gt_flat[idx]
    coords = np.array([(i // w, i % w) for i in idx])
    return patches, labels, coords, h, w

class PatchAutoencoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

class SimpleTransformer(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        z = z.unsqueeze(1)
        attn_out, _ = self.attn(z, z, z)
        squeezed = attn_out.squeeze(1)
        scores = self.linear(squeezed).squeeze()
        return scores

def visualize_latent_space(z, labels, dataset_name, output_dir):
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', n_iter=500)
    z_2d = tsne.fit_transform(z)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab20', s=5)
    plt.title(f"Latent Space t-SNE Visualization - {dataset_name.upper()}")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_tsne_visualization.png")
    plt.close()

class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def run_pipeline_with_files(hsi_path, gt_path, dataset_name, patch_size=16, latent_dim=32, num_epochs=10, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(hsi_path))
    os.makedirs(output_dir, exist_ok=True)
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = sio.loadmat(hsi_path)
    gt = sio.loadmat(gt_path)

    data_keys = [key for key in data.keys() if not key.startswith('__')]
    gt_keys = [key for key in gt.keys() if not key.startswith('__')]

    if not data_keys or not gt_keys:
        raise ValueError("No valid variable keys found in .mat files")

    data_array = data[data_keys[0]]
    gt_array = gt[gt_keys[0]]

    print("Loading dataset and preprocessing...")
    data_pca, gt_processed, h, w, pca_dim = preprocess(data_array, gt_array, dataset_name)

    rgb_image = data_pca[:, :, :3]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    rgb_image_path = os.path.join(output_dir, f"{dataset_name}_pca_rgb.png")
    plt.imsave(rgb_image_path, rgb_image)

    input_dim = patch_size * patch_size * pca_dim
    patches, labels, coords, h, w = extract_patches(data_pca, gt_processed, patch_size=patch_size)

    print(f"Extracted {len(patches)} patches with known labels (no redundancy).")

    patches_tensor = torch.tensor(patches, dtype=torch.float32).reshape(-1, input_dim)
    dataset = TensorDataset(patches_tensor)
    batch_size = 512
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = PatchAutoencoder(latent_dim=latent_dim, input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=3)

    print("Training Autoencoder with mixed precision and batching...")
    model.train()
    epoch_losses = []

    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output, _ = model(batch)
                loss = criterion(output, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.6f}")

    loss_curve_path = os.path.join(output_dir, f"{dataset_name}_ae_loss_curve.png")
    plt.figure()
    plt.plot(range(1, num_epochs+1), epoch_losses, marker='o')
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(loss_curve_path)
    plt.close()

    print("Extracting latent features...")
    model.eval()
    with torch.no_grad():
        _, latent_z = model(patches_tensor.to(device))
    latent_z = latent_z.cpu()

    print("Visualizing latent space with optimized t-SNE...")
    visualize_latent_space(latent_z.numpy(), labels, dataset_name, output_dir)

    print("Running Transformer with batched scoring...")
    transformer = SimpleTransformer(dim=latent_dim).to(device)
    transformer.eval()
    trans_scores = []
    for i in range(0, latent_z.shape[0], batch_size):
        batch = latent_z[i:i+batch_size].to(device)
        with torch.no_grad():
            scores = transformer(batch).cpu().numpy()
        trans_scores.append(scores)
    trans_scores = np.concatenate(trans_scores)

    print("Creating anomaly map...")
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    for idx, (x, y) in enumerate(coords):
        anomaly_map[x, y] = trans_scores[idx]
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    rgb_image = data_pca[:, :, :3]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    axs[0].imshow(gt_processed, cmap='tab20')
    axs[0].set_title("Ground Truth")
    axs[0].axis('off')

    axs[1].imshow(rgb_image)
    axs[1].set_title("RGB PCA Image")
    axs[1].axis('off')

    im = axs[2].imshow(anomaly_map_norm, cmap='inferno')
    axs[2].set_title("Anomaly Heatmap")
    axs[2].axis('off')
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

    axs[3].imshow(rgb_image)
    axs[3].imshow(anomaly_map_norm, cmap='inferno', alpha=0.4)
    axs[3].set_title("Overlay RGB + Anomalies")
    axs[3].axis('off')

    plt.tight_layout()
    overlay_path = os.path.join(output_dir, f"{dataset_name}_anomaly_map_overlay.png")
    plt.savefig(overlay_path)
    plt.close()

    # Save anomaly score map as image
    anomaly_map_path = os.path.join(output_dir, f"{dataset_name}_anomaly_map.png")
    plt.imsave(anomaly_map_path, anomaly_map_norm, cmap='inferno')

    print("Training SVM on PCA-reduced latent features...")
    pca_svm = PCA(n_components=min(latent_dim, 20))
    latent_reduced = pca_svm.fit_transform(latent_z.numpy())

    X_train, X_test, y_train, y_test = train_test_split(latent_reduced, labels, test_size=0.25, random_state=42, stratify=labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in zip(np.unique(y_train), class_weights)}

    svm_clf = SVC(kernel='rbf', C=5, gamma='scale', class_weight=class_weight_dict, probability=True)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    y_proba = svm_clf.predict_proba(X_test)

    classification_report_str = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc = None
    ap = None
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        ap = average_precision_score(y_test, y_proba, average='macro')
    except Exception as e:
        classification_report_str += f"\nAUC/AP metrics failed: {str(e)}"

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {dataset_name.upper()}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    conf_matrix_path = os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()

    print("Pipeline complete! Results saved in output directory.")

    results = {
        'stats': {
            'accuracy': accuracy,
            'classification_report': classification_report_str,
            'auc': auc,
            'average_precision': ap
        },
        'images': [
            {
                'url': f'/uploads/{dataset_name}_confusion_matrix.png',
                'name': 'Confusion Matrix',
                'description': 'Visualization of model predictions vs true labels'
            },
            {
                'url': f'/uploads/{dataset_name}_tsne_visualization.png',
                'name': 't-SNE Visualization',
                'description': '2D visualization of the latent space'
            },
            {
                'url': f'/uploads/{dataset_name}_anomaly_map.png',
                'name': 'Anomaly Score Map',
                'description': 'Spatial distribution of anomaly scores'
            },
            {
                'url': f'/uploads/{dataset_name}_pca_rgb.png',
                'name': 'PCA RGB Image',
                'description': 'RGB image from PCA components'
            },
            {
                'url': f'/uploads/{dataset_name}_ae_loss_curve.png',
                'name': 'Autoencoder Loss Curve',
                'description': 'Training loss curve of the autoencoder'
            },
            {
                'url': f'/uploads/{dataset_name}_anomaly_map_overlay.png',
                'name': 'Anomaly Map Overlay',
                'description': 'Overlay of RGB image and anomaly heatmap'
            }
        ],
        'info': f'Analysis completed for {dataset_name} dataset with {len(labels)} samples'
    }

    return results

