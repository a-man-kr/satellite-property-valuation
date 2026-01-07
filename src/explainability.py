"""
Model Explainability Tools
Grad-CAM and feature importance visualization for property valuation models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path


class GradCAM:
    """
    Grad-CAM implementation for CNN explainability.
    Highlights regions in satellite images that influence price predictions.
    """
    
    def __init__(self, model: nn.Module, target_layer_name: str = 'layer4'):
        """
        Initialize Grad-CAM.
        
        Args:
            model: CNN model (e.g., ResNet)
            target_layer_name: Name of the layer to visualize
        """
        self.model = model
        self.model.eval()
        
        # Get target layer
        self.target_layer = self._get_layer(target_layer_name)
        
        # Storage
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _get_layer(self, layer_name: str):
        """Get layer by name from model."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor: torch.Tensor, target_output: torch.Tensor = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_output: Target for backpropagation (uses model output if None)
        
        Returns:
            Heatmap as numpy array
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        if target_output is None:
            target_output = output
        target_output.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def visualize(self, image_path: str, transform, output_path: str = None,
                  alpha: float = 0.4) -> plt.Figure:
        """
        Generate and visualize Grad-CAM overlay.
        
        Args:
            image_path: Path to input image
            transform: Image preprocessing transform
            output_path: Path to save visualization (optional)
            alpha: Overlay transparency
        
        Returns:
            Matplotlib figure
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Preprocess
        input_tensor = transform(img).unsqueeze(0)
        
        # Generate CAM
        cam = self.generate(input_tensor)
        
        # Resize to image size
        cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = ((1 - alpha) * img_array + alpha * heatmap).astype(np.uint8)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_array)
        axes[0].set_title('Original Satellite Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Areas Influencing Price)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig


class FeatureImportanceAnalyzer:
    """Analyze and visualize feature importance for property valuation."""
    
    def __init__(self, feature_names: list):
        self.feature_names = feature_names
    
    def plot_importance(self, importance_scores: np.ndarray, top_k: int = 15,
                        title: str = "Feature Importance", output_path: str = None):
        """
        Plot feature importance bar chart.
        
        Args:
            importance_scores: Array of importance scores
            top_k: Number of top features to show
            title: Plot title
            output_path: Path to save figure
        """
        # Sort by importance
        indices = np.argsort(importance_scores)[::-1][:top_k]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importance_scores[indices], color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def compare_importance(self, tabular_importance: np.ndarray, 
                          multimodal_importance: np.ndarray,
                          output_path: str = None):
        """
        Compare feature importance between tabular and multimodal models.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # Tabular model
        indices_tab = np.argsort(tabular_importance)[::-1][:10]
        axes[0].barh(range(10), tabular_importance[indices_tab], color='gray')
        axes[0].set_yticks(range(10))
        axes[0].set_yticklabels([self.feature_names[i] for i in indices_tab])
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Tabular Model')
        
        # Multimodal model
        indices_mm = np.argsort(multimodal_importance)[::-1][:10]
        axes[1].barh(range(10), multimodal_importance[indices_mm], color='steelblue')
        axes[1].set_yticks(range(10))
        axes[1].set_yticklabels([self.feature_names[i] for i in indices_mm])
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Multimodal Model')
        
        plt.suptitle('Feature Importance Comparison')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig


def analyze_visual_features(image_path: str, output_path: str = None):
    """
    Analyze visual features in satellite image that may affect property value.
    
    Features analyzed:
    - Green coverage (vegetation)
    - Water presence
    - Road/pavement density
    - Building density
    """
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Green detection (vegetation)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / green_mask.size
    
    # Blue detection (water)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    water_ratio = np.sum(blue_mask > 0) / blue_mask.size
    
    # Gray detection (roads/pavement)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray_mask = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY)
    road_ratio = np.sum((gray_mask > 100) & (gray_mask < 200)) / gray_mask.size
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(green_mask, cmap='Greens')
    axes[0, 1].set_title(f'Vegetation: {green_ratio*100:.1f}%')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(blue_mask, cmap='Blues')
    axes[1, 0].set_title(f'Water: {water_ratio*100:.1f}%')
    axes[1, 0].axis('off')
    
    # Summary
    axes[1, 1].bar(['Vegetation', 'Water', 'Roads'], 
                   [green_ratio*100, water_ratio*100, road_ratio*100],
                   color=['green', 'blue', 'gray'])
    axes[1, 1].set_ylabel('Coverage (%)')
    axes[1, 1].set_title('Visual Feature Summary')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return {
        'vegetation_ratio': green_ratio,
        'water_ratio': water_ratio,
        'road_ratio': road_ratio
    }


if __name__ == "__main__":
    # Example usage
    print("Explainability module loaded successfully")
