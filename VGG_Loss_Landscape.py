from data.loaders import get_cifar_loader
from models.vgg import VGG_A_BatchNorm, VGG_A
from IPython import display
from tqdm import tqdm as tqdm
import random
import os
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import pandas as pd
mpl.use('Agg')

# Set font support for English
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]

# ## Constants (parameters) initialization
device_id = [0, 1, 2, 3]
num_workers = 0  # Set to 0 for Windows to avoid multiprocessing issues
batch_size = 128

# add our package dir to path
module_path = os.path.dirname(os.getcwd())
home_path = module_path

# Generate timestamp to avoid file overwriting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
figures_path = os.path.join(home_path, 'reports', 'figures', timestamp)
models_path = os.path.join(home_path, 'reports', 'models', timestamp)
losses_path = os.path.join(home_path, 'reports', 'losses', timestamp)
grads_path = os.path.join(home_path, 'reports', 'grads', timestamp)

# Ensure directories for saving losses and gradients exist
os.makedirs(losses_path, exist_ok=True)
os.makedirs(grads_path, exist_ok=True)

# Make sure you are using the right device.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
if use_cuda:
    print(f"CUDA : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"device {i}: {torch.cuda.get_device_name(i)}")

# This function is used to calculate the accuracy of model classification


def get_accuracy(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

# Set a random seed to ensure reproducible results


def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training


def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None, lr=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)

            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()

            # Calculate gradient
            loss.backward()

            # Record gradient (fix: calculate gradient magnitude correctly)
            current_grad_value = None
            if hasattr(model, 'classifier') and len(model.classifier) > 4:
                weight_grad = model.classifier[4].weight.grad
                if weight_grad is not None:
                    # Calculate mean absolute value of the gradient
                    current_grad_value = np.mean(
                        np.abs(weight_grad.clone().cpu().numpy()))

            # Append the current gradient value to the list
            if current_grad_value is not None:
                grad.append(current_grad_value)

            optimizer.step()
            optimizer.zero_grad()  # Clear gradients for next iteration

        losses_list.append(loss_list)
        grads.append(grad)

        # Calculate accuracies
        train_acc = get_accuracy(model, train_loader)
        val_acc = get_accuracy(model, val_loader)
        train_accuracy_curve[epoch] = train_acc
        val_accuracy_curve[epoch] = val_acc

        # Save best model (with complete information)
        if val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            max_val_accuracy_epoch = epoch
            if best_model_path is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_acc,
                }, best_model_path)

        print(f'Epoch {epoch+1}/{epochs_n}')
        print(f'Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')
        print(
            f'Best Val Accuracy: {max_val_accuracy:.4f} at epoch {max_val_accuracy_epoch+1}')

    # Save losses and gradients
    np.save(os.path.join(losses_path, f'losses_lr_{lr}.npy'), losses_list)
    np.save(os.path.join(grads_path, f'grads_lr_{lr}.npy'), grads)

    return losses_list, grads, train_accuracy_curve, val_accuracy_curve

# Helper function: Align training data of different lengths and compute statistics


def align_and_compute_statistics(data_list, max_length=None, percentile_range=(0, 100)):
    """
    Align training data of different lengths and compute statistical metrics

    Args:
    data_list: A list containing data from multiple training epochs, 
               each element is a list of loss or gradient values for an epoch
    max_length: Maximum length. If None, use the maximum length in the data
    percentile_range: Percentile range, tuple (lower bound, upper bound)

    Returns:
    x: Array of time steps
    min_curve: Minimum value curve for each time step
    max_curve: Maximum value curve for each time step
    mean_curve: Mean value curve for each time step
    """
    if max_length is None:
        max_length = max(len(data) for data in data_list)

    # Create a 2D array filled with NaN for missing values
    aligned_data = np.full((len(data_list), max_length), np.nan)
    for i, data in enumerate(data_list):
        aligned_data[i, :len(data)] = data

    # Compute statistics for each time step
    x = np.arange(max_length)
    min_curve = np.nanpercentile(aligned_data, percentile_range[0], axis=0)
    max_curve = np.nanpercentile(aligned_data, percentile_range[1], axis=0)
    mean_curve = np.nanmean(aligned_data, axis=0)

    return x, min_curve, max_curve, mean_curve

# Plot loss landscape using improved statistical methods


def plot_loss_landscape(losses_without_bn, losses_with_bn, title, save_path,
                        percentile_range=(5, 95), plot_mean=True):
    """
    Plot loss landscape with linear y-axis scale
    """
    # Compute aligned statistics
    x, min_without_bn, max_without_bn, mean_without_bn = align_and_compute_statistics(
        losses_without_bn, percentile_range=percentile_range)
    x, min_with_bn, max_with_bn, mean_with_bn = align_and_compute_statistics(
        losses_with_bn, percentile_range=percentile_range)

    plt.figure(figsize=(12, 7))

    # Plot losses without BN
    plt.fill_between(x, min_without_bn, max_without_bn, color='blue', alpha=0.2,
                     label=f'Loss Range without BN ({percentile_range[0]}-{percentile_range[1]}%)')
    if plot_mean:
        plt.plot(x, mean_without_bn, color='blue', linestyle='-',
                 label='Mean Loss without BN', alpha=0.7)

    # Plot losses with BN
    plt.fill_between(x, min_with_bn, max_with_bn, color='red', alpha=0.2,
                     label=f'Loss Range with BN ({percentile_range[0]}-{percentile_range[1]}%)')
    if plot_mean:
        plt.plot(x, mean_with_bn, color='red', linestyle='-',
                 label='Mean Loss with BN', alpha=0.7)

    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Remove scientific notation and use linear scale
    plt.ticklabel_format(axis='y', style='plain', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_gradient_landscape(grads_without_bn, grads_with_bn, title, save_path,
                            percentile_range=(5, 95), plot_mean=True):
    """
    Plot gradient landscape with linear y-axis scale
    """
    # Compute aligned statistics
    x, min_without_bn, max_without_bn, mean_without_bn = align_and_compute_statistics(
        grads_without_bn, percentile_range=percentile_range)
    x, min_with_bn, max_with_bn, mean_with_bn = align_and_compute_statistics(
        grads_with_bn, percentile_range=percentile_range)

    plt.figure(figsize=(12, 7))

    # Plot gradients without BN
    plt.fill_between(x, min_without_bn, max_without_bn, color='blue', alpha=0.2,
                     label=f'Gradient Range without BN ({percentile_range[0]}-{percentile_range[1]}%)')
    if plot_mean:
        plt.plot(x, mean_without_bn, color='blue', linestyle='-',
                 label='Mean Gradient without BN', alpha=0.7)

    # Plot gradients with BN
    plt.fill_between(x, min_with_bn, max_with_bn, color='red', alpha=0.2,
                     label=f'Gradient Range with BN ({percentile_range[0]}-{percentile_range[1]}%)')
    if plot_mean:
        plt.plot(x, mean_with_bn, color='red', linestyle='-',
                 label='Mean Gradient with BN', alpha=0.7)

    plt.xlabel('Training Steps')
    plt.ylabel('Gradient Value')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Remove scientific notation and use linear scale
    plt.ticklabel_format(axis='y', style='plain', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_accuracy_comparison(accuracies, title, save_path):
    """
    Plot accuracy comparison between different models and learning rates

    Args:
    accuracies: Dictionary containing accuracy curves for different models and learning rates
    title: Plot title
    save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 7))

    colors = ['blue', 'red', 'green', 'purple']

    for i, (lr, lr_data) in enumerate(accuracies.items()):
        # Plot training accuracy
        plt.plot(lr_data['train_without_bn'], color=colors[i], linestyle='-',
                 label=f'Train w/o BN, lr={lr}')
        plt.plot(lr_data['train_with_bn'], color=colors[i], linestyle='--',
                 label=f'Train with BN, lr={lr}')

        # Plot validation accuracy
        plt.plot(lr_data['val_without_bn'], color=colors[i], linestyle=':',
                 label=f'Val w/o BN, lr={lr}')
        plt.plot(lr_data['val_with_bn'], color=colors[i], linestyle='-.',
                 label=f'Val with BN, lr={lr}')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    # Windows system multiprocessing support
    if os.name == 'nt':
        import torch.multiprocessing as mp
        mp.freeze_support()

    # Create output directories with timestamp
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    # Initialize data loaders
    try:
        train_loader = get_cifar_loader(train=True, num_workers=num_workers)
        val_loader = get_cifar_loader(train=False, num_workers=num_workers)

        # Verify data loader
        X, y = next(iter(train_loader))
        print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        exit(1)

    # Training parameters
    epo = 20
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]

    all_losses_without_bn = []
    all_grads_without_bn = []
    all_losses_with_bn = []
    all_grads_with_bn = []

    # Dictionary to store accuracy curves for each learning rate
    accuracy_data = {}

    for lr in learning_rates:
        set_random_seeds(seed_value=2020, device=str(device))
        accuracy_data[lr] = {}

        # Train model without BN
        model_without_bn = VGG_A()
        optimizer_without_bn = torch.optim.Adam(
            model_without_bn.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_model_path_without_bn = os.path.join(
            models_path, f'best_model_without_bn_{lr}_{timestamp}.pth')
        losses_without_bn, grads_without_bn, train_acc_without_bn, val_acc_without_bn = train(
            model_without_bn, optimizer_without_bn, criterion, train_loader, val_loader, epochs_n=epo, best_model_path=best_model_path_without_bn, lr=lr
        )
        all_losses_without_bn.extend(losses_without_bn)
        all_grads_without_bn.extend(grads_without_bn)

        # Store accuracy curves
        accuracy_data[lr]['train_without_bn'] = train_acc_without_bn
        accuracy_data[lr]['val_without_bn'] = val_acc_without_bn

        # Train model with BN
        model_with_bn = VGG_A_BatchNorm()
        optimizer_with_bn = torch.optim.Adam(model_with_bn.parameters(), lr=lr)
        best_model_path_with_bn = os.path.join(
            models_path, f'best_model_with_bn_{lr}_{timestamp}.pth')
        losses_with_bn, grads_with_bn, train_acc_with_bn, val_acc_with_bn = train(
            model_with_bn, optimizer_with_bn, criterion, train_loader, val_loader, epochs_n=epo, best_model_path=best_model_path_with_bn, lr=lr
        )
        all_losses_with_bn.extend(losses_with_bn)
        all_grads_with_bn.extend(grads_with_bn)

        # Store accuracy curves
        accuracy_data[lr]['train_with_bn'] = train_acc_with_bn
        accuracy_data[lr]['val_with_bn'] = val_acc_with_bn

    # Plot loss landscape for each learning rate
    for lr in learning_rates:
        # Filter losses for current learning rate
        lr_losses_without_bn = all_losses_without_bn[(
            learning_rates.index(lr)*epo):((learning_rates.index(lr)+1)*epo)]
        lr_losses_with_bn = all_losses_with_bn[(
            learning_rates.index(lr)*epo):((learning_rates.index(lr)+1)*epo)]

        plot_loss_landscape(lr_losses_without_bn, lr_losses_with_bn,
                            f'Loss Landscape Comparison (w/o BN vs w/ BN), lr={lr}',
                            os.path.join(figures_path, f'loss_landscape_comparison_{lr}_{timestamp}.png'))

        # Filter gradients for current learning rate
        lr_grads_without_bn = all_grads_without_bn[(
            learning_rates.index(lr)*epo):((learning_rates.index(lr)+1)*epo)]
        lr_grads_with_bn = all_grads_with_bn[(
            learning_rates.index(lr)*epo):((learning_rates.index(lr)+1)*epo)]

        plot_gradient_landscape(lr_grads_without_bn, lr_grads_with_bn,
                                f'Gradient Landscape Comparison (w/o BN vs w/ BN), lr={lr}',
                                os.path.join(figures_path, f'gradient_landscape_comparison_{lr}_{timestamp}.png'))

    # Plot accuracy comparison for all learning rates
    plot_accuracy_comparison(accuracy_data,
                             'Accuracy Comparison between models with/without BatchNorm',
                             os.path.join(figures_path, f'accuracy_comparison_{timestamp}.png'))
