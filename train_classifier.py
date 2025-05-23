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
import time
from datetime import datetime, timedelta
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

    # Print debug information
    print(f"[Rank {args.rank}] Initializing process group")
    print(f"[Rank {args.rank}] world_size: {args.world_size}")
    print(f"[Rank {args.rank}] rank: {args.rank}")
    print(f"[Rank {args.rank}] local_rank: {args.local_rank}")
    print(f"[Rank {args.rank}] master_addr: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"[Rank {args.rank}] master_port: {os.environ.get('MASTER_PORT', 'Not set')}")

    args.distributed = True
    
    # Set the device before initializing process group
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device(f'cuda:{args.local_rank}')
    
    # Set PyTorch/NCCL environment variables for better stability
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_NTHREADS'] = '4'
    os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
    os.environ['NCCL_MIN_NCHANNELS'] = '4'
    
    # Try to automatically detect the network interface
    try:
        import socket
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        interface_name = None
        
        import netifaces
        for iface in netifaces.interfaces():
            try:
                if netifaces.ifaddresses(iface)[netifaces.AF_INET][0]['addr'] == ip:
                    interface_name = iface
                    break
            except (KeyError, IndexError):
                continue
        
        if interface_name:
            print(f"[Rank {args.rank}] Detected network interface: {interface_name}")
            os.environ['NCCL_SOCKET_IFNAME'] = interface_name
        else:
            print(f"[Rank {args.rank}] Warning: Could not detect network interface automatically")
    except ImportError:
        print(f"[Rank {args.rank}] Warning: netifaces package not found, skipping automatic interface detection")
    except Exception as e:
        print(f"[Rank {args.rank}] Warning: Error detecting network interface: {str(e)}")
    
    # Initialize process group with a timeout and device_id
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"[Rank {args.rank}] Attempting to initialize process group (attempt {retry_count + 1}/{max_retries})")
            
            # Set initialization parameters
            init_params = {
                'backend': args.dist_backend,
                'init_method': 'env://',
                'timeout': timedelta(minutes=2),
                'world_size': args.world_size,
                'rank': args.rank,
            }
            
            dist.init_process_group(**init_params)
            
            # Initialize process group
            print(f"[Rank {args.rank}] Successfully initialized process group")
            
            # Verify the process group is working
            if dist.is_initialized():
                print(f"[Rank {args.rank}] Process group verification successful")
                # Try a simple all_reduce to verify communication
                test_tensor = torch.tensor([args.rank], device=args.device)
                dist.all_reduce(test_tensor)
                print(f"[Rank {args.rank}] Initial all_reduce test successful")
            break
            
        except Exception as e:
            retry_count += 1
            print(f"[Rank {args.rank}] Failed to initialize process group (attempt {retry_count}/{max_retries})")
            print(f"[Rank {args.rank}] Error: {str(e)}")
            if retry_count == max_retries:
                raise
            time.sleep(10)  # Wait before retrying
    
    # Add barrier with device_ids and timeout
    try:
        dist.barrier(device_ids=[args.local_rank], timeout=timedelta(seconds=60))
    except Exception as e:
        print(f"[Rank {args.rank}] Warning: Initial barrier synchronization failed: {str(e)}")
        print(f"[Rank {args.rank}] Attempting to continue anyway...")

def cleanup_distributed():
    """Cleanup distributed training resources."""
    if dist.is_initialized():
        try:
            # Try to synchronize before destroying
            dist.barrier()
            dist.destroy_process_group()
            print("Successfully cleaned up distributed training resources")
        except Exception as e:
            print(f"Warning: Error during distributed cleanup: {str(e)}")
            # Force cleanup even if synchronization fails
            dist.destroy_process_group()

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, args, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)
    
    batch_times = []
    data_times = []
    forward_times = []
    backward_times = []
    
    end = time.time()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', disable=not args.rank == 0)
    
    # Add synchronization points
    sync_period = 10  # Synchronize every 10 batches
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # Periodic synchronization
        if args.distributed and batch_idx % sync_period == 0:
            try:
                # Simple barrier without timeout
                dist.barrier(device_ids=[args.local_rank])
            except Exception as e:
                print(f"[Rank {args.rank}] Warning: Periodic sync failed at batch {batch_idx}: {str(e)}")
        
        data_time = time.time() - end
        data_times.append(data_time)
        
        try:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Forward pass timing
            forward_start = time.time()
            outputs = model(inputs)
            torch.cuda.synchronize()
            forward_time = time.time() - forward_start
            forward_times.append(forward_time)
            
            # Handle multiple outputs during training (fixed point correction)
            if isinstance(outputs, list):
                loss = sum(criterion(output, targets) for output in outputs) / len(outputs)
                outputs = outputs[-1]  # Use last output for accuracy calculation
            else:
                loss = criterion(outputs, targets)
            
            # Backward pass timing
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        except RuntimeError as e:
            print(f"[Rank {args.rank}] Error in training batch {batch_idx}: {str(e)}")
            if "NCCL" in str(e):
                print(f"[Rank {args.rank}] Attempting to recover from NCCL error...")
                try:
                    dist.barrier(device_ids=[args.local_rank])
                except:
                    pass
                continue
            else:
                raise e
        
        # Batch time
        batch_time = time.time() - end
        batch_times.append(batch_time)
        end = time.time()
        
        # Update progress bar on main process
        if args.rank == 0:
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total,
                'batch_time': f'{np.mean(batch_times):.3f}s',
                'data_time': f'{np.mean(data_times):.3f}s'
            })
    
    scheduler.step()
    
    # Final synchronization
    if args.distributed:
        try:
            dist.barrier(device_ids=[args.local_rank])
        except Exception as e:
            print(f"[Rank {args.rank}] Warning: Final epoch sync failed: {str(e)}")
    
    # Synchronize metrics across processes
    if args.distributed:
        try:
            # Convert to tensors on GPU
            metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=device)
            
            # All-reduce without timeout
            dist.all_reduce(metrics)
            
            total_loss, correct, total = metrics.tolist()
            total_loss = total_loss / dist.get_world_size()
            
        except Exception as e:
            print(f"[Rank {args.rank}] Warning: Metric synchronization failed: {str(e)}")
            # Use local metrics if synchronization fails
            pass
    
    timing_stats = {
        'batch_time': np.mean(batch_times),
        'data_time': np.mean(data_times),
        'forward_time': np.mean(forward_times),
        'backward_time': np.mean(backward_times)
    }
    
    return total_loss / len(train_loader), 100. * correct / total, timing_stats

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
    
    # Download dataset only on rank 0
    if args.rank == 0:
        print("Rank 0: Downloading CIFAR-10 dataset...")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform_test)
        print("Rank 0: Dataset download complete")
    
    # Make sure all processes wait for the data to be downloaded
    if args.distributed:
        dist.barrier()
    
    # Now all processes can load the dataset
    if args.rank != 0:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=False, transform=transform_test)
    
    if args.distributed:
        train_sampler = DistributedSampler(trainset, shuffle=True)
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
        # Enable finding unused parameters for DDP
        model = DDP(model, 
                   device_ids=[args.local_rank],
                   find_unused_parameters=True,
                   broadcast_buffers=False)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args.warmup_epochs, args.epochs)
    
    # Training loop with better formatting
    best_acc = 0
    total_train_time = 0
    
    if args.rank == 0:
        print("\n" + "="*80)
        print("Starting Training".center(80))
        print(f"Total epochs: {args.epochs}".center(80))
        print(f"Batches per epoch: {len(trainloader)}".center(80))
        print(f"Batch size per GPU: {args.batch_size}".center(80))
        print(f"Total batch size: {args.batch_size * (dist.get_world_size() if args.distributed else 1)}".center(80))
        print("="*80 + "\n")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        if args.rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}".center(80, '-'))
            print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
            print('-'*80)
        
        # Train
        train_loss, train_acc, timing_stats = train_epoch(model, trainloader, criterion, 
                                                         optimizer, scheduler, device, args, epoch)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, testloader, criterion, device, args)
        
        # Timing information
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        avg_epoch_time = total_train_time / (epoch + 1)
        estimated_remaining = avg_epoch_time * (args.epochs - epoch - 1)
        
        if args.rank == 0:
            # Print epoch summary with better formatting
            print('\n' + '-'*80)
            print('Epoch Summary'.center(80))
            print('-'*80)
            
            # Training metrics
            print('\nTraining:')
            print(f"Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
            
            # Testing metrics
            print('\nValidation:')
            print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")
            
            # Timing information
            print('\nTiming Information:')
            print(f"Epoch time: {str(timedelta(seconds=int(epoch_time)))}")
            print(f"Average batch time: {timing_stats['batch_time']:.3f}s")
            print(f"Average data time: {timing_stats['data_time']:.3f}s")
            print(f"Average forward time: {timing_stats['forward_time']:.3f}s")
            print(f"Average backward time: {timing_stats['backward_time']:.3f}s")
            print(f"Estimated time remaining: {str(timedelta(seconds=int(estimated_remaining)))}")
            
            # Save checkpoint if improved
            if test_acc > best_acc:
                print('\nCheckpoint:')
                print(f"Best accuracy improved from {best_acc:.2f}% to {test_acc:.2f}%")
                state = {
                    'model': model.module.state_dict() if args.distributed else model.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                    'args': args,
                }
                torch.save(state, f'{args.output_dir}/best_classifier.pth')
                best_acc = test_acc
            
            print('-'*80 + '\n')
    
    if args.rank == 0:
        print("="*80)
        print("Training Complete".center(80))
        print(f"Total training time: {str(timedelta(seconds=int(total_train_time)))}".center(80))
        print(f"Average epoch time: {str(timedelta(seconds=int(avg_epoch_time)))}".center(80))
        print(f"Best test accuracy: {best_acc:.2f}%".center(80))
        print("="*80 + "\n")
    
    # Cleanup
    cleanup_distributed()

if __name__ == '__main__':
    main() 