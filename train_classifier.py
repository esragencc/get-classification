import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
from tqdm import tqdm
from equilibrium_transformer_classifier import GETClassifier
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def get_args():
    parser = argparse.ArgumentParser(description='Train GET Classifier on CIFAR-10')
    
    # Distributed training parameters
    parser.add_argument('--world-size', default=-1, type=int,
                      help='Number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                      help='Node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                      help='URL used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                      help='Distributed backend')
    parser.add_argument('--local_rank', type=int, default=-1,
                      help='Local rank for distributed training')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=384,
                      help='Hidden dimension size')
    parser.add_argument('--deq_depth', type=int, default=3,
                      help='Number of DEQ blocks')
    parser.add_argument('--num_heads', type=int, default=6,
                      help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                      help='MLP expansion ratio')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                      help='Weight decay for optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                      help='Number of warmup epochs')
    
    # DEQ parameters
    parser.add_argument('--f_solver', type=str, default='anderson',
                      help='Fixed point solver type')
    parser.add_argument('--f_thres', type=float, default=30,
                      help='Maximum number of fixed point iterations')
    parser.add_argument('--f_eps', type=float, default=1e-5,
                      help='Fixed point convergence threshold')
    parser.add_argument('--f_solver_eps', type=float, default=1e-5,
                      help='Solver epsilon')
    parser.add_argument('--stop_mode', type=str, default='rel',
                      help='Stopping criterion mode')
    parser.add_argument('--anderson_m', type=int, default=5,
                      help='Anderson memory size')
    parser.add_argument('--beta', type=float, default=1.0,
                      help='Beta parameter for solver')
    
    # Device and output parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    return args

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

def setup_distributed(args):
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank})', flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    dist.barrier()

def cleanup_distributed():
    """Cleanup distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)
    
    pbar = tqdm(train_loader, desc='Training', disable=not args.rank == 0)
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
        
        # Update progress bar on main process
        if args.rank == 0:
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    scheduler.step()
    
    # Synchronize metrics across processes
    if args.distributed:
        total_loss = torch.tensor(total_loss).cuda()
        correct = torch.tensor(correct).cuda()
        total = torch.tensor(total).cuda()
        
        dist.all_reduce(total_loss)
        dist.all_reduce(correct)
        dist.all_reduce(total)
        
        total_loss = total_loss.item() / dist.get_world_size()
        correct = correct.item()
        total = total.item()
    
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device, args):
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
    
    # Synchronize metrics across processes
    if args.distributed:
        total_loss = torch.tensor(total_loss).cuda()
        correct = torch.tensor(correct).cuda()
        total = torch.tensor(total).cuda()
        
        dist.all_reduce(total_loss)
        dist.all_reduce(correct)
        dist.all_reduce(total)
        
        total_loss = total_loss.item() / dist.get_world_size()
        correct = correct.item()
        total = total.item()
    
    return total_loss / len(test_loader), 100. * correct / total

def main():
    # Get arguments and set seed
    args = get_args()
    
    # Setup distributed training
    setup_distributed(args)
    
    # Set device and seed
    if args.distributed:
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(args.seed + args.rank if args.distributed else args.seed)
    
    # Create output directory (only on main process)
    if args.rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
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
                                          download=args.rank == 0, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=args.rank == 0, transform=transform_test)
    
    if args.distributed:
        train_sampler = DistributedSampler(trainset)
        test_sampler = DistributedSampler(testset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
    
    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=2,
        sampler=train_sampler,
        pin_memory=True
    )
    
    testloader = DataLoader(
        testset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        sampler=test_sampler,
        pin_memory=True
    )
    
    # Create model
    model = GETClassifier(
        args=args,
        hidden_size=args.hidden_size,
        deq_depth=args.deq_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio
    ).to(device)
    
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args.warmup_epochs, args.epochs)
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        if args.rank == 0:
            print(f'\nEpoch: {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, trainloader, criterion, 
                                          optimizer, scheduler, device, args)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, testloader, criterion, device, args)
        
        if args.rank == 0:
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}%')
            print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
            
            # Save checkpoint
            if test_acc > best_acc:
                print(f'Saving checkpoint... Best accuracy improved from {best_acc:.3f} to {test_acc:.3f}')
                state = {
                    'model': model.module.state_dict() if args.distributed else model.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                    'args': args,
                }
                torch.save(state, f'{args.output_dir}/best_classifier.pth')
                best_acc = test_acc
    
    # Cleanup
    cleanup_distributed()

if __name__ == '__main__':
    main() 