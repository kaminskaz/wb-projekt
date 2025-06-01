import clip
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from types import MethodType
import os
import zipfile
import requests
from io import BytesIO
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Store activations
activations = {}

# Wrap ViT blocks to capture intermediate [CLS] outputs
def wrap_vit_blocks_dino(model):
    activations.clear()
    original_blocks = model.blocks  

    for i, block in enumerate(original_blocks):
        def make_custom_forward(orig_forward, layer_name):
            def custom_forward(self, x):
                out = orig_forward(x)
                activations[layer_name] = out.detach()
                return out
            return custom_forward

        block.forward = MethodType(make_custom_forward(block.forward, f"layer_{i}"), block)

    return activations

# Logit Lens Analysis with Cosine + Predictions
def logit_lens_analysis_dino(activations, projection_head, final_output, temperature=1.0):
    distances = {}
    predictions = {}

    for name, x in activations.items():
        cls_token = x[:, 0, :]  # CLS token
        projected = projection_head(cls_token)  # (batch, num_classes)

        # Cosine similarity to final CLS token
        projected_norm = F.normalize(projected, dim=-1)
        final_norm = F.normalize(final_output[:, 0, :], dim=-1)
        similarity = F.cosine_similarity(projected_norm, final_norm, dim=-1)
        distances[name] = similarity.detach().cpu().item()

        # Top-1 prediction and probability
        probs = F.softmax(projected / temperature, dim=-1)
        top_prob, top_class = torch.max(probs, dim=-1)
        predictions[f"{name}_label"] = int(top_class[0].cpu().item())
        predictions[f"{name}_prob"] = float(top_prob[0].cpu().item())

    return distances, predictions

# Run on dataset and save CSV logs
def perform_logit_lens_analysis(model, dataset, device,
                                cosine_path="logit_lens_results/cosine_similarity.csv",
                                preds_path="logit_lens_results/predictions.csv"):
    model.eval()
    os.makedirs("logit_lens_results", exist_ok=True)

    wrap_vit_blocks_dino(model)
    headers = [f"layer_{i}" for i in range(len(model.blocks))]

    for image_idx, (image, label) in enumerate(dataset):
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            final_output = model.forward_features(image) 
            final_output = F.normalize(final_output, dim=-1)

            distances, predictions = logit_lens_analysis_dino(
                activations,
                model.head,
                final_output
            )

        # Save cosine similarity
        cosine_header = ['Image'] + headers
        write_header = not os.path.exists(cosine_path) or os.path.getsize(cosine_path) == 0
        with open(cosine_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(cosine_header)
            cosine_row = [f"Image_{image_idx + 1}"] + [distances[layer] for layer in headers]
            writer.writerow(cosine_row)

        # Save top-1 predictions and probabilities
        pred_header = ['Image'] + [f"{layer}_label" for layer in headers] + [f"{layer}_prob" for layer in headers]
        write_header = not os.path.exists(preds_path) or os.path.getsize(preds_path) == 0
        with open(preds_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(pred_header)
            pred_labels = [predictions[f"{layer}_label"] for layer in headers]
            pred_probs = [predictions[f"{layer}_prob"] for layer in headers]
            pred_row = [f"Image_{image_idx + 1}"] + pred_labels + pred_probs
            writer.writerow(pred_row)

# Plot Cosine Similarity and Prediction Probabilities
def plot_results(distances, predictions):
    layer_names = sorted(
        [k for k in distances.keys() if k.startswith("layer_")],
        key=lambda x: int(x.split('_')[1])
    )

    try:
        similarity_values = [float(distances[layer]) for layer in layer_names]
        prob_values = [float(predictions.get(f"{layer}_prob", np.nan)) for layer in layer_names]
        predicted_labels = [predictions.get(f"{layer}_label", "") for layer in layer_names]
    except Exception as e:
        print("Error preparing data:", e)
        return

    if not all(isinstance(v, (int, float)) for v in similarity_values + prob_values):
        print("Some values are not numeric.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1]})

    sns.heatmap(np.array(similarity_values).reshape(1, -1), annot=True, cmap="viridis",
                xticklabels=layer_names, yticklabels=["Cosine Similarity"], cbar=True,
                ax=axes[0], cbar_kws={'label': 'Cosine Similarity'})

    sns.heatmap(np.array(prob_values).reshape(1, -1), annot=True, cmap="magma",
                xticklabels=layer_names, yticklabels=["Prediction Prob."], cbar=True,
                ax=axes[1], cbar_kws={'label': 'Prediction Probability'})

    for i, label in enumerate(predicted_labels):
        if label is not None:
            axes[1].text(i + 0.5, -0.3, str(label), ha='center', va='center',
                         color='black', fontsize=9, rotation=90,
                         transform=axes[1].transData)

    plt.suptitle("Cosine Similarity & Prediction Probability per Layer", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
extract_path = "./tiny-imagenet-200"

print("Downloading Tiny ImageNet...")
response = requests.get(url)
with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(".")

print("Download and extraction complete.")


train_dir = os.path.join(extract_path, "train")
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
])
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


import torch
import torch.nn as nn
import torch.optim as optim
import timm

# Config
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 200

# Load DINO ViT
model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)

# Freeze all backbone params
for param in model.parameters():
    param.requires_grad = False

# Replace the head
num_features = model.num_features  # <--- FIXED: works for ViT with identity head
model.head = nn.Linear(num_features, num_classes)

# Enable training for new head
for param in model.head.parameters():
    param.requires_grad = True

# Send to device
model = model.to(device)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from torchvision import datasets, transforms

# ----------------------------
# Config
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 200
batch_size = 32
epochs = 10
lr = 0.001

# ----------------------------
# Data (use your own transforms if already defined)
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

train_dataset = datasets.ImageFolder('tiny-imagenet-200/train', transform=train_transform)
val_dataset = datasets.ImageFolder('tiny-imagenet-200/test', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ----------------------------
# Model: ViT DINO + Linear Head
# ----------------------------
model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace head
model.head = nn.Linear(model.num_features, num_classes)
for param in model.head.parameters():
    param.requires_grad = True

model = model.to(device)

# ----------------------------
# Training setup
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=lr)

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = total_loss / total
    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f}")

    # Optional: Evaluation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1) 
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total
    print(f"Validation Acc: {val_acc:.4f}")


perform_logit_lens_analysis(model=model, dataset=train_dataset, device=device, cosine_path="logit_lens_results/DINO/cosine_similarity.csv")

