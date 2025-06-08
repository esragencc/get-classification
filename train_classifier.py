import argparse
import os
import time

import numpy as np
from glob import glob

import torch
# Enable optimizations for faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import logging

from models.get import GET_Classifier_models

from torchdeq import add_deq_args
from torchdeq.loss import fp_correction


def create_logger(logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        if logging_dir:
            logging.basicConfig(
                level=logging.INFO,
                format='[\033[34m%(asctime)s\033[0m] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='[\033[34m%(asctime)s\033[0m] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler()]
            )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main(args):
    '''
    CIFAR-10 Classification training with GET.
    '''
    # Setup DDP
    dist.init_process_group('nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert args.global_batch_size % world_size == 0, f'Batch size must be divisible by world size.'
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f'Starting rank={rank}, seed={seed}, world_size={world_size}.')

    # Setup an experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f'{args.results_dir}/*'))
        model_string_name = args.model.replace('/', '-')
        experiment_dir = f'{args.results_dir}/{experiment_index:03d}-{model_string_name}-{args.name}'

        checkpoint_dir = f'{experiment_dir}/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f'Experiment directory created at {experiment_dir}')
    else:
        logger = create_logger()

    # Create model
    model = GET_Classifier_models[args.model](
        args=args,
        input_size=args.input_size,
        num_classes=args.num_classes
    )
    
    # Setup DDP
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    logger.info(f'Model Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Setup learning rate scheduler with extended warmup
    warmup_steps = 1000
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine annealing after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Setup data
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

    # Training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform_train
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=transform_test
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.eval_batch_size // world_size),
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info(f'Training dataset contains {len(train_dataset):,} images')
    logger.info(f'Test dataset contains {len(test_dataset):,} images')

    # Prepare model for training
    model.train()
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Variables for monitoring/logging purposes
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_acc1 = 0
    running_acc5 = 0
    total_steps = args.epochs * len(train_loader)

    # Resume from checkpoint if specified
    if args.resume:
        ckpt = torch.load(args.resume, map_location=torch.device('cpu'))
        model.module.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        scheduler.load_state_dict(ckpt['scheduler'])
        train_steps = max(args.resume_iter, 0)
        logger.info(f'Resume from {args.resume}..')

    start_time = time.time()
    logger.info(f'Training for {args.epochs} epochs...')
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        logger.info(f'Beginning epoch {epoch}...')
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Handle DEQ output (list during training)
            if isinstance(output, list):
                # Check for NaN in DEQ outputs
                for i, out in enumerate(output):
                    if torch.isnan(out).any():
                        logger.error(f"NaN detected in DEQ iteration {i} at step {train_steps}!")
                        logger.error(f"Input stats: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
                        raise ValueError("DEQ output contains NaN")
                
                # Use fp_correction for DEQ training
                loss, loss_list = fp_correction(criterion, (output, target), return_loss_values=True)
                # Use final output for accuracy computation
                final_output = output[-1]
            else:
                # Check for NaN in regular output
                if torch.isnan(output).any():
                    logger.error(f"NaN detected in model output at step {train_steps}!")
                    logger.error(f"Input stats: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
                    raise ValueError("Model output contains NaN")
                
                loss = criterion(output, target)
                final_output = output
            
            # Check for NaN loss
            if torch.isnan(loss):
                logger.error(f"NaN loss detected at step {train_steps}!")
                logger.error(f"Output stats: min={final_output.min():.4f}, max={final_output.max():.4f}, mean={final_output.mean():.4f}")
                raise ValueError("Training diverged with NaN loss")
            
            # Compute accuracy
            acc1, acc5 = accuracy(final_output, target, topk=(1, 5))
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            opt.step()
            scheduler.step()  # Step scheduler every iteration

            # Update running statistics
            if isinstance(output, list):
                running_loss += loss_list[-1]
            else:
                running_loss += loss.item()
            running_acc1 += acc1.item()
            running_acc5 += acc5.item()
            log_steps += 1
            train_steps += 1

            # Log training progress
            if train_steps % args.log_every == 0:
                # Measure training speed
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss and accuracy over all processes
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_acc1 = torch.tensor(running_acc1 / log_steps, device=device)
                avg_acc5 = torch.tensor(running_acc5 / log_steps, device=device)
                
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_acc1, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_acc5, op=dist.ReduceOp.SUM)
                
                avg_loss = avg_loss.item() / world_size
                avg_acc1 = avg_acc1.item() / world_size
                avg_acc5 = avg_acc5.item() / world_size
                
                logger.info(f'(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, '
                           f'Acc@1: {avg_acc1:.2f}%, Acc@5: {avg_acc5:.2f}%, '
                           f'Steps/Sec: {steps_per_sec:.2f}')

                # Reset monitoring variables
                running_loss = 0
                running_acc1 = 0
                running_acc5 = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint_path = f'{checkpoint_dir}/{train_steps:07d}.pth'
                    save_ckpt_classifier(args, model, opt, scheduler, checkpoint_path)
                    logger.info(f'Saved checkpoint to {checkpoint_path}')
                dist.barrier()

            # Evaluation
            if train_steps % args.eval_every == 0 and train_steps > 0:
                test_acc1, test_acc5, test_loss = evaluate(model, test_loader, criterion, device, world_size)
                if rank == 0:
                    logger.info(f'(step={train_steps:07d}) Test Loss: {test_loss:.4f}, '
                               f'Test Acc@1: {test_acc1:.2f}%, Test Acc@5: {test_acc5:.2f}%')
                model.train()  # Switch back to training mode
                dist.barrier()

    # Final evaluation
    test_acc1, test_acc5, test_loss = evaluate(model, test_loader, criterion, device, world_size)
    if rank == 0:
        logger.info(f'Final Test Results - Loss: {test_loss:.4f}, '
                   f'Acc@1: {test_acc1:.2f}%, Acc@5: {test_acc5:.2f}%')
        
        # Save final checkpoint
        checkpoint_path = f'{checkpoint_dir}/final.pth'
        save_ckpt_classifier(args, model, opt, scheduler, checkpoint_path)
        logger.info(f'Saved final checkpoint to {checkpoint_path}')
    
    dist.barrier()
    dist.destroy_process_group()


def evaluate(model, test_loader, criterion, device, world_size):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    test_acc1 = 0
    test_acc5 = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Handle DEQ output
            if isinstance(output, list):
                output = output[-1]  # Use final iteration for evaluation
            
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            test_loss += loss.item() * data.size(0)
            test_acc1 += acc1.item() * data.size(0)
            test_acc5 += acc5.item() * data.size(0)
            total_samples += data.size(0)
    
    # Reduce across all processes
    test_loss_tensor = torch.tensor(test_loss, device=device)
    test_acc1_tensor = torch.tensor(test_acc1, device=device)
    test_acc5_tensor = torch.tensor(test_acc5, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    
    dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(test_acc1_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(test_acc5_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    
    test_loss = test_loss_tensor.item() / total_samples_tensor.item()
    test_acc1 = test_acc1_tensor.item() / total_samples_tensor.item()
    test_acc5 = test_acc5_tensor.item() / total_samples_tensor.item()
    
    return test_acc1, test_acc5, test_loss


def save_ckpt_classifier(args, model, opt, scheduler, checkpoint_path):
    """Save checkpoint for classifier"""
    torch.save({
        'model': model.module.state_dict(),
        'opt': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args': args
    }, checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--name', type=str, default='cifar10_classification')
    parser.add_argument('--results_dir', type=str, default='results_classifier')

    parser.add_argument('--model', type=str, choices=list(GET_Classifier_models.keys()), default='GET-Classifier-S')
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=10)
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
 
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--global_batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--global_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--mem', action='store_true', help='Enable O(1) memory.')

    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--ckpt_every', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=1000)

    parser.add_argument('--resume', help="restore checkpoint for training")
    parser.add_argument('--resume_iter', type=int, default=-1, help="resume from the given iterations")

    # Add DEQ args
    add_deq_args(parser)

    args = parser.parse_args()
    main(args) 