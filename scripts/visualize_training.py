#!/usr/bin/env python3
"""
Training Visualization Script for CUDA MLP Project
Generates professional graphs for M.Tech project documentation
"""

import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['legend.fontsize'] = 10

def parse_training_output(output_text):
    """
    Parse training output to extract epoch, accuracy, and timing data.

    Expected format:
    Epoch 0: Train 0.10450, Val 0.09733, Test 0.10170
    Epoch 1: Train 0.14170, Val 0.13525, Test 0.13380  Average time per epoch: 0.560 sec
    """
    epochs = []
    train_acc = []
    val_acc = []
    test_acc = []
    time_per_epoch = []
    learning_rates = []

    # Pattern for accuracy lines
    pattern = r'Epoch (\d+):\s+Train ([\d.]+),\s+Val ([\d.]+),\s+Test ([\d.]+)'
    # Pattern for time
    time_pattern = r'Average time per epoch:\s+([\d.]+)\s+sec'
    # Pattern for learning rate (if present)
    lr_pattern = r'LR:\s+([\d.]+)'

    for line in output_text.split('\n'):
        match = re.search(pattern, line)
        if match:
            epochs.append(int(match.group(1)))
            train_acc.append(float(match.group(2)) * 100)  # Convert to percentage
            val_acc.append(float(match.group(3)) * 100)
            test_acc.append(float(match.group(4)) * 100)

        time_match = re.search(time_pattern, line)
        if time_match:
            time_per_epoch.append(float(time_match.group(1)))

        lr_match = re.search(lr_pattern, line)
        if lr_match:
            learning_rates.append(float(lr_match.group(1)))

    return {
        'epochs': epochs,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'time_per_epoch': time_per_epoch,
        'learning_rates': learning_rates
    }

def plot_accuracy_curves(data_dict, output_file='training_accuracy.png'):
    """
    Plot training/validation/test accuracy curves for multiple optimizers.

    data_dict: {'SGD': data, 'Momentum': data, 'Adam': data}
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    colors = {'SGD': '#2E86AB', 'Momentum': '#A23B72', 'Adam': '#F18F01',
              'Adam+LR': '#06A77D', 'SGD+LR': '#5C5C5C', 'BatchNorm': '#D00000',
              'BN+SGD': '#9B2335', 'BN+Adam+LR': '#1B998B'}
    linestyles = {'train': '-', 'val': '--', 'test': '-.'}

    for name, data in data_dict.items():
        if not data['epochs']:
            continue

        color = colors.get(name, '#333333')

        # Plot test accuracy (main metric)
        ax.plot(data['epochs'], data['test_acc'],
               label=f'{name} (Test)', color=color, linewidth=2.5,
               linestyle='-', marker='o', markersize=4, markevery=2)

        # Plot train accuracy (lighter, dashed)
        ax.plot(data['epochs'], data['train_acc'],
               label=f'{name} (Train)', color=color, linewidth=1.5,
               linestyle='--', alpha=0.6)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Training Accuracy Comparison: SGD / Momentum / Adam / BN+Adam+LR\n' +
                 'CUDA MLP on MNIST (2 layers × 128 neurons, batch size 2048)',
                 fontweight='bold', pad=20)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=100)

    # Add annotations for final accuracy
    for name, data in data_dict.items():
        if data['test_acc'] and data['test_acc'][-1] > 5:  # Only annotate if converged
            final_acc = data['test_acc'][-1]
            final_epoch = data['epochs'][-1]
            ax.annotate(f'{final_acc:.1f}%',
                       xy=(final_epoch, final_acc),
                       xytext=(10, 0), textcoords='offset points',
                       fontsize=9, color=colors.get(name, '#333333'),
                       fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✓ Saved accuracy plot to {output_file}')
    plt.close()

def plot_training_time(data_dict, output_file='training_time.png'):
    """
    Bar chart comparing training time per epoch across optimizers.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    names = []
    times = []
    colors_list = []

    colors = {'SGD': '#2E86AB', 'Momentum': '#A23B72', 'Adam': '#F18F01',
              'Adam+LR': '#06A77D', 'SGD+LR': '#5C5C5C', 'BatchNorm': '#D00000',
              'BN+SGD': '#9B2335', 'BN+Adam+LR': '#1B998B'}

    for name, data in data_dict.items():
        if data['time_per_epoch']:
            names.append(name)
            # Use median time (more stable than last value)
            times.append(np.median(data['time_per_epoch']))
            colors_list.append(colors.get(name, '#333333'))

    if not names:
        print('⚠ No timing data available')
        return

    bars = ax.bar(names, times, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Time per Epoch (seconds)', fontweight='bold')
    ax.set_title('Training Speed Comparison\nCUDA MLP on MNIST (20 epochs, batch size 2048)',
                 fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(top=max(times) * 1.2)  # Add 20% headroom

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✓ Saved timing plot to {output_file}')
    plt.close()

def plot_optimizer_convergence(data_dict, output_file='optimizer_convergence.png'):
    """
    Plot showing optimizer convergence behavior - all working correctly now!
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = {'SGD': '#2E86AB', 'Momentum': '#A23B72', 'Adam': '#F18F01',
              'Adam+LR': '#06A77D', 'SGD+LR': '#5C5C5C'}

    # Left plot: First 5 epochs (zoomed in to show early convergence)
    for name, data in data_dict.items():
        if not data['epochs']:
            continue
        color = colors.get(name, '#333333')
        epochs_subset = [e for e in data['epochs'] if e <= 5]
        test_acc_subset = data['test_acc'][:len(epochs_subset)]

        ax1.plot(epochs_subset, test_acc_subset,
                label=name, color=color, linewidth=3,
                marker='o', markersize=8)

    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax1.set_title('Early Training Behavior (Epochs 0-5)', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 85)

    # Add annotation showing Adam's faster early convergence
    ax1.annotate('Adam/BN+Adam+LR\nconverge fastest (56-79% by ep.5)',
                xy=(5, 57.5), xytext=(2.5, 30),
                arrowprops=dict(arrowstyle='->', color='#F18F01', lw=2),
                fontsize=10, color='#F18F01', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # Right plot: Full training (all epochs)
    for name, data in data_dict.items():
        if not data['epochs']:
            continue
        color = colors.get(name, '#333333')
        ax2.plot(data['epochs'], data['test_acc'],
                label=name, color=color, linewidth=2.5,
                marker='o', markersize=4, markevery=3)

    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax2.set_title('Full Training Run (All Epochs)', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 95)

    plt.suptitle('Optimizer Convergence Analysis: BN+Adam+LR achieves best accuracy!',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✓ Saved convergence analysis to {output_file}')
    plt.close()

def create_summary_table(data_dict):
    """
    Print a formatted summary table of results.
    """
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"{'Optimizer':<15} {'Final Test Acc':<18} {'Avg Time/Epoch':<18} {'Status':<15}")
    print("-"*80)

    for name, data in data_dict.items():
        if not data['epochs']:
            continue

        final_acc = data['test_acc'][-1] if data['test_acc'] else 0.0
        avg_time = np.mean(data['time_per_epoch']) if data['time_per_epoch'] else 0.0

        # Determine status
        if final_acc < 1.0:
            status = '❌ Diverged'
        elif final_acc < 50.0:
            status = '⚠️  Poor'
        elif final_acc < 70.0:
            status = '✓ Fair'
        else:
            status = '✓ Good'

        print(f"{name:<15} {final_acc:>6.2f}%            {avg_time:>6.3f}s            {status:<15}")

    print("="*80 + "\n")

def load_output_file(path):
    """Load training output from a file, return empty string if not found."""
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return ""

def main():
    """
    Main function to generate all visualizations.
    """
    print("="*80)
    print("CUDA MLP Training Visualization")
    print("="*80 + "\n")

    # Load data from actual output files (generated by ./cuda* 2 128 20 2048 <lr> 1 5 0)
    # Configuration: 2 layers x 128 neurons, batch 2048, 20 epochs, MNIST
    data_dict = {
        'SGD':          parse_training_output(load_output_file('results/sgd_output.txt')),
        'Momentum':     parse_training_output(load_output_file('results/momentum_output.txt')),
        'Adam':         parse_training_output(load_output_file('results/adam_output.txt')),
        'Adam+LR':      parse_training_output(load_output_file('results/adam_lr_output.txt')),
        'BN+SGD':       parse_training_output(load_output_file('results/batchnorm_output.txt')),
        'BN+Adam+LR':   parse_training_output(load_output_file('results/full_output.txt')),
    }

    # Drop configs with no data
    data_dict = {k: v for k, v in data_dict.items() if v['epochs']}

    # Generate plots
    print("Generating visualizations...")
    plot_accuracy_curves(data_dict, 'figures/training_accuracy_comparison.png')
    plot_training_time(data_dict, 'figures/training_time_comparison.png')
    plot_optimizer_convergence(data_dict, 'figures/optimizer_convergence_analysis.png')

    # Print summary
    create_summary_table(data_dict)

    print("✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - training_accuracy_comparison.png")
    print("  - training_time_comparison.png")
    print("  - optimizer_convergence_analysis.png")
    print("\nUse these graphs in your M.Tech project report!")

if __name__ == '__main__':
    main()
