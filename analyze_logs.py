import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def parse_log_file(filepath):
    """
    Parses the training log file to extract training and testing metrics.
    """
    train_data = []
    test_data = []

    train_log_pattern = re.compile(
        r"\(step=(\d+)\) Train Loss: ([\d.]+), Acc@1: ([\d.]+)%, Acc@5: ([\d.]+)%"
    )
    test_log_pattern = re.compile(
        r"\(step=(\d+)\) Test Loss: ([\d.]+), Test Acc@1: ([\d.]+)%, Test Acc@5: ([\d.]+)%"
    )

    with open(filepath, 'r') as f:
        for line in f:
            train_match = train_log_pattern.search(line)
            if train_match:
                step = int(train_match.group(1))
                loss = float(train_match.group(2))
                acc1 = float(train_match.group(3))
                acc5 = float(train_match.group(4))
                train_data.append([step, loss, acc1, acc5])
                continue

            test_match = test_log_pattern.search(line)
            if test_match:
                step = int(test_match.group(1))
                loss = float(test_match.group(2))
                acc1 = float(test_match.group(3))
                acc5 = float(test_match.group(4))
                test_data.append([step, loss, acc1, acc5])

    train_df = pd.DataFrame(train_data, columns=['Step', 'Loss', 'Acc@1', 'Acc@5'])
    test_df = pd.DataFrame(test_data, columns=['Step', 'Loss', 'Acc@1', 'Acc@5'])

    return train_df, test_df

def plot_metrics(train_df, test_df, output_dir='plots'):
    """
    Generates and saves plots for training and testing metrics.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Loss (Train vs. Test)
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['Step'], train_df['Loss'], label='Train Loss', alpha=0.8)
    plt.plot(test_df['Step'], test_df['Loss'], label='Test Loss', marker='o', linestyle='--')
    plt.title('Training and Test Loss vs. Steps')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_vs_steps.png'))
    plt.close()
    print(f"Saved loss plot to {os.path.join(output_dir, 'loss_vs_steps.png')}")

    # Plot 2: Accuracy (Train vs. Test)
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['Step'], train_df['Acc@1'], label='Train Acc@1', color='b', alpha=0.8)
    plt.plot(test_df['Step'], test_df['Acc@1'], label='Test Acc@1', color='b', marker='o', linestyle='--')
    
    plt.plot(train_df['Step'], train_df['Acc@5'], label='Train Acc@5', color='r', alpha=0.8)
    plt.plot(test_df['Step'], test_df['Acc@5'], label='Test Acc@5', color='r', marker='o', linestyle='--')
    
    plt.title('Training and Test Accuracy vs. Steps')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_steps.png'))
    plt.close()
    print(f"Saved accuracy plot to {os.path.join(output_dir, 'accuracy_vs_steps.png')}")

def main():
    parser = argparse.ArgumentParser(description="Analyze a CIFAR-10 training log file and generate plots.")
    parser.add_argument('log_file', type=str, help="Path to the log file (e.g., cifar10.txt)")
    parser.add_argument('--output_dir', type=str, default='training_plots', help="Directory to save the plots.")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found at {args.log_file}")
        return

    train_df, test_df = parse_log_file(args.log_file)
    
    if train_df.empty or test_df.empty:
        print("Could not find training or testing data in the log file.")
        return
        
    plot_metrics(train_df, test_df, args.output_dir)

if __name__ == '__main__':
    main() 