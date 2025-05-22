import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchdeq import get_deq
import argparse
from tqdm import tqdm

from equilibrium_transformer_classifier import EquilibriumClassifier

def get_args():
    parser = argparse.ArgumentParser(description='Train Equilibrium Transformer on CIFAR')
    
    # Model parameters
    parser.add_argument('--hidden-size', type=int, default=384,
                        help='Hidden dimension size')
    parser.add_argument('--num-heads', type=int, default=6,
                        help='Number of attention heads')
    parser.add_argument('--deq-depth', type=int, default=3,
                        help='Number of DEQ blocks')
    parser.add_argument('--patch-size', type=int, default=4,
                        help='Size of image patches')
    parser.add_argument('--mlp-ratio', type=float, default=4.0,
                        help='MLP expansion ratio')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs')
    
    # DEQ parameters
    parser.add_argument('--f-thres', type=int, default=24,
                        help='Forward threshold for DEQ solver')
    parser.add_argument('--b-thres', type=int, default=24,
                        help='Backward threshold for DEQ solver')
    parser.add_argument('--mem', action='store_true',
                        help='Use memory efficient computation')
    
    # Other parameters
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-dir', default='checkpoints',
                        help='Directory to save checkpoints')
    
    return parser.parse_args()

def get_cifar10_loaders(args):
    # Data augmentation and normalization for training
    # Just normalization for validation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    valset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val)
    valloader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return trainloader, valloader

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, trainloader, criterion, optimizer, scheduler, deq_solver, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(trainloader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, deq_solver)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': total_loss/(batch_idx+1),
            'acc': 100.*correct/total
        })
    
    return total_loss/len(trainloader), 100.*correct/total

@torch.no_grad()
def evaluate(model, valloader, criterion, deq_solver, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(valloader, desc='Evaluating'):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs, deq_solver)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss/len(valloader), 100.*correct/total

def main():
    args = get_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Get data loaders
    trainloader, valloader = get_cifar10_loaders(args)
    
    # Create model
    model = EquilibriumClassifier(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        deq_depth=args.deq_depth,
        patch_size=args.patch_size,
        mlp_ratio=args.mlp_ratio,
        mem_efficient=args.mem
    ).to(device)
    
    # Setup DEQ solver
    deq_args = argparse.Namespace(
        f_thres=args.f_thres,
        b_thres=args.b_thres,
        stop_mode="rel",
        eps=1e-4
    )
    deq_solver = get_deq(deq_args)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Calculate total steps for scheduler
    total_steps = len(trainloader) * args.epochs
    warmup_steps = len(trainloader) * args.warmup_epochs
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch+1}')
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, scheduler, deq_solver, device
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(
            model, valloader, criterion, deq_solver, device
        )
        
        # Save checkpoint if best accuracy
        if val_acc > best_acc:
            print('Saving checkpoint...')
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(args.save_dir, 'best_model.pth'))
        
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}%')
        print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}%')

if __name__ == '__main__':
    main() 