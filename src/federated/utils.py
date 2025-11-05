"""
Utility Functions for FLAIR-X
Data loading, model weight conversion, training helpers
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import OrderedDict
import numpy as np
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, BATCH_SIZE, IMAGE_SIZE, DEVICE

# ============================================
# Data Loading
# ============================================

def get_data_transforms():
    """Get data augmentation and normalization transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def load_data(hospital_id, split="train"):
    """
    Load data for a specific hospital and split
    
    Args:
        hospital_id: "A", "B", or "C"
        split: "train", "val", or "test"
    
    Returns:
        DataLoader
    """
    train_transform, test_transform = get_data_transforms()
    transform = train_transform if split == "train" else test_transform
    
    data_path = DATA_DIR / f"hospital_{hospital_id}" / split
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    dataset = datasets.ImageFolder(str(data_path), transform=transform)
    
    # Shuffle only for training
    shuffle = (split == "train")
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )
    
    return loader


def get_all_data_loaders(hospital_id):
    """Get train, val, and test loaders for a hospital"""
    return {
        "train": load_data(hospital_id, "train"),
        "val": load_data(hospital_id, "val"),
        "test": load_data(hospital_id, "test")
    }


# ============================================
# Model Weight Conversion (PyTorch ↔ NumPy)
# ============================================

def get_parameters(model):
    """Extract model parameters as NumPy arrays (for Flower)"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """Set model parameters from NumPy arrays (from Flower)"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# ============================================
# Training & Evaluation
# ============================================

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """Train for one epoch with error handling"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    try:
        for batch_idx, (data, target) in enumerate(data_loader):
            try:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"⚠️ NaN loss detected at batch {batch_idx}", file=sys.stderr)
                    continue
                
                loss.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
            except Exception as batch_error:
                print(f"❌ Error in batch {batch_idx}: {batch_error}", file=sys.stderr)
                raise
        
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
        
    except Exception as e:
        print(f"❌ Error in train_one_epoch: {e}", file=sys.stderr)
        raise


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / total
    test_acc = correct / total
    
    return test_loss, test_acc


# ============================================
# Data Statistics
# ============================================

def get_dataset_stats(hospital_id):
    """Get statistics about hospital's dataset"""
    stats = {}
    
    for split in ["train", "val", "test"]:
        data_path = DATA_DIR / f"hospital_{hospital_id}" / split
        
        if not data_path.exists():
            stats[split] = {"total": 0, "class_distribution": {}}
            continue
        
        dataset = datasets.ImageFolder(str(data_path))
        
        class_counts = {}
        for _, label in dataset.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        stats[split] = {
            "total": len(dataset),
            "class_distribution": class_counts
        }
    
    return stats


if __name__ == "__main__":
    # Test data loading
    print("🧪 Testing data loaders...\n")
    
    for hospital_id in ["A", "B", "C"]:
        print(f"📊 Hospital {hospital_id}:")
        try:
            stats = get_dataset_stats(hospital_id)
            for split, info in stats.items():
                print(f"   {split}: {info['total']} images")
                if info['class_distribution']:
                    print(f"      Class distribution: {info['class_distribution']}")
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
        print()