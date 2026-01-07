"""
CNN Encoder for Satellite Image Feature Extraction
Uses pretrained ResNet18 to extract visual embeddings from satellite imagery.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm


class SatelliteImageEncoder(nn.Module):
    """CNN encoder for extracting features from satellite images."""
    
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        
        # Get the feature dimension from ResNet
        backbone_out_features = self.backbone.fc.in_features  # 512 for ResNet18
        
        # Replace final FC layer with embedding layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(backbone_out_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        """Extract embeddings from images."""
        return self.backbone(x)
    
    def get_transform(self):
        """Get image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class ImageFeatureExtractor:
    """Extract and cache image features for all properties."""
    
    def __init__(self, model: SatelliteImageEncoder, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.transform = model.get_transform()
        
    def extract_single(self, image_path: str) -> np.ndarray:
        """Extract features from a single image."""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return np.zeros(self.model.embedding_dim)
    
    def extract_batch(self, property_ids: list, image_dir: str, 
                      batch_size: int = 32) -> dict:
        """Extract features for multiple properties."""
        image_dir = Path(image_dir)
        features_dict = {}
        
        # Collect valid images
        valid_ids = []
        valid_paths = []
        
        for pid in property_ids:
            img_path = image_dir / f"{pid}.png"
            if img_path.exists():
                valid_ids.append(pid)
                valid_paths.append(img_path)
        
        print(f"Found {len(valid_ids)}/{len(property_ids)} images")
        
        # Process in batches
        for i in tqdm(range(0, len(valid_ids), batch_size)):
            batch_ids = valid_ids[i:i+batch_size]
            batch_paths = valid_paths[i:i+batch_size]
            
            # Load and transform images
            batch_tensors = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    batch_tensors.append(self.transform(img))
                except:
                    batch_tensors.append(torch.zeros(3, 224, 224))
            
            batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                batch_features = self.model(batch)
            
            for pid, feat in zip(batch_ids, batch_features.cpu().numpy()):
                features_dict[pid] = feat
        
        # Fill missing with zeros
        for pid in property_ids:
            if pid not in features_dict:
                features_dict[pid] = np.zeros(self.model.embedding_dim)
        
        return features_dict
    
    def save_features(self, features_dict: dict, output_path: str):
        """Save extracted features to file."""
        np.savez(output_path, **{str(k): v for k, v in features_dict.items()})
    
    def load_features(self, input_path: str) -> dict:
        """Load features from file."""
        data = np.load(input_path)
        return {k: data[k] for k in data.files}


class GradCAMVisualizer:
    """Grad-CAM visualization for model explainability."""
    
    def __init__(self, model: nn.Module, target_layer: str = 'layer4'):
        self.model = model
        self.model.eval()
        
        # Get target layer
        self.target_layer = dict(model.backbone.named_modules())[target_layer]
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Generate Grad-CAM heatmap for an image."""
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Backward pass (use output as target for regression)
        output.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def visualize(self, image_path: str, transform, save_path: str = None):
        """Generate and optionally save Grad-CAM visualization."""
        import matplotlib.pyplot as plt
        import cv2
        
        # Load and process image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        img_tensor = transform(img).unsqueeze(0)
        
        # Generate CAM
        cam = self.generate_cam(img_tensor)
        
        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = (0.6 * img_array + 0.4 * heatmap).astype(np.uint8)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Test the encoder
    encoder = SatelliteImageEncoder(embedding_dim=256)
    print(f"Encoder output dimension: {encoder.embedding_dim}")
    
    # Test with random input
    x = torch.randn(4, 3, 224, 224)
    out = encoder(x)
    print(f"Output shape: {out.shape}")
