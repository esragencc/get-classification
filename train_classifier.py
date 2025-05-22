import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from equilibrium_transformer_classifier import GETClassifier

# Default configuration
config = {
    # Model parameters
    'hidden_size': 384,
    'deq_depth': 3,
    'num_heads': 6,
    'mlp_ratio': 4.0,
    
    # Training parameters
    'batch_size': 128,
    'epochs': 200,
    'lr': 0.001,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    
    # DEQ parameters
    'f_solver': 'anderson',
    'f_thres': 30,
    'f_eps': 1e-5,
    'f_solver_eps': 1e-5,
    'stop_mode': 'rel',
    'anderson_m': 5,
    'beta': 1.0,
    
    # Device and output parameters
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'checkpoints',
    'seed': 42
}

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_scheduler(optimizer, warmup_epochs, epochs):
    """Get cosine scheduler with warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * float(epoch - warmup_epochs) / float(epochs - warmup_epochs)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Handle multiple outputs during training (fixed point correction)
        if isinstance(outputs, list):
            loss = sum(criterion(output, targets) for output in outputs) / len(outputs)
            outputs = outputs[-1]  # Use last output for accuracy calculation
        else:
            loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    scheduler.step()
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(test_loader), 100. * correct / total

def main():
    # Set seed and create output directory
    set_seed(config['seed'])
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=config['batch_size'],
                           shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=config['batch_size'],
                          shuffle=False, num_workers=2)
    
    # Create model
    model = GETClassifier(
        args=config,
        hidden_size=config['hidden_size'],
        deq_depth=config['deq_depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio']
    ).to(config['device'])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = get_scheduler(optimizer, config['warmup_epochs'], config['epochs'])
    
    # Training loop
    best_acc = 0
    for epoch in range(config['epochs']):
        print(f'\nEpoch: {epoch+1}/{config["epochs"]}')
        
        # Train
        train_loss, train_acc = train_epoch(model, trainloader, criterion, 
                                          optimizer, scheduler, config['device'])
        
        # Evaluate
        test_loss, test_acc = evaluate(model, testloader, criterion, config['device'])
        
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}%')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
        
        # Save checkpoint
        if test_acc > best_acc:
            print(f'Saving checkpoint... Best accuracy improved from {best_acc:.3f} to {test_acc:.3f}')
            state = {
                'model': model.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
                'config': config,
            }
            torch.save(state, f'{config["output_dir"]}/best_classifier.pth')
            best_acc = test_acc

if __name__ == '__main__':
    main() 