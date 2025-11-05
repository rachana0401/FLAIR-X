"""
Model Definition for Diabetic Retinopathy Classification
5 classes: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR
"""
import torch
import torch.nn as nn
from torchvision import models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import NUM_CLASSES, MODEL_NAME

class DRModel(nn.Module):
    """Diabetic Retinopathy Classification Model"""
    
    def __init__(self, pretrained=True):
        super(DRModel, self).__init__()
        
        if MODEL_NAME == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, NUM_CLASSES)
        
        elif MODEL_NAME == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, NUM_CLASSES)
        
        else:
            raise ValueError(f"Unknown model: {MODEL_NAME}")
    
    def forward(self, x):
        return self.backbone(x)


def get_model():
    """Factory function to create model"""
    return DRModel(pretrained=True)


if __name__ == "__main__":
    # Test the model
    model = get_model()
    print(f"✅ Model created: {MODEL_NAME}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"   Output shape: {output.shape} (batch_size, {NUM_CLASSES} classes)")
    print(f"   Output logits: {output[0].detach().numpy()}")