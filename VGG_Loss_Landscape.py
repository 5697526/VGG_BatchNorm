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

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# ## Constants (parameters) initialization
device_id = [0, 1, 2, 3]
num_workers = 0  # Set to 0 for Windows to avoid multiprocessing issues
batch_size = 128

# add our package dir to path
module_path = os.path.dirname(os.getcwd())
home_path = module_path

# 生成时间戳以避免文件覆盖
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
figures_path = os.path.join(home_path, 'reports', 'figures', timestamp)
models_path = os.path.join(home_path, 'reports', 'models', timestamp)
losses_path = os.path.join(home_path, 'reports', 'losses', timestamp)
grads_path = os.path.join(home_path, 'reports', 'grads', timestamp)

# 确保保存损失和梯度的目录存在
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
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 5))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve[:epoch+1])
        axes[0].set_title('Training Loss')

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

        axes[1].plot(train_accuracy_curve[:epoch+1], label='Train')
        axes[1].plot(val_accuracy_curve[:epoch+1], label='Validation')
        axes[1].set_title('Accuracy')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(
            figures_path, f'training_progress_lr_{lr}_epoch{epoch+1}.png'), dpi=300)
        plt.close()  # Prevent memory leak

        print(f'Epoch {epoch+1}/{epochs_n}')
        print(f'Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')
        print(
            f'Best Val Accuracy: {max_val_accuracy:.4f} at epoch {max_val_accuracy_epoch+1}')

    # 保存损失和梯度
    np.save(os.path.join(losses_path, f'losses_lr_{lr}.npy'), losses_list)
    np.save(os.path.join(grads_path, f'grads_lr_{lr}.npy'), grads)

    return losses_list, grads

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()


def plot_loss_landscape(losses_without_bn, losses_with_bn, title, save_path):
    min_curve_without_bn = np.min(losses_without_bn, axis=0)
    max_curve_without_bn = np.max(losses_without_bn, axis=0)
    min_curve_with_bn = np.min(losses_with_bn, axis=0)
    max_curve_with_bn = np.max(losses_with_bn, axis=0)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(min_curve_without_bn))

    plt.plot(x, min_curve_without_bn,
             label='Min Loss without BN', color='blue')
    plt.plot(x, max_curve_without_bn, label='Max Loss without BN',
             color='blue', linestyle='--')
    plt.fill_between(x, min_curve_without_bn,
                     max_curve_without_bn, color='blue', alpha=0.2)

    plt.plot(x, min_curve_with_bn, label='Min Loss with BN', color='red')
    plt.plot(x, max_curve_with_bn, label='Max Loss with BN',
             color='red', linestyle='--')
    plt.fill_between(x, min_curve_with_bn, max_curve_with_bn,
                     color='red', alpha=0.2)

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

# Plot gradient landscape


def plot_gradient_landscape(grads_without_bn, grads_with_bn, title, save_path):
    min_curve_without_bn = np.min(grads_without_bn, axis=0)
    max_curve_without_bn = np.max(grads_without_bn, axis=0)
    min_curve_with_bn = np.min(grads_with_bn, axis=0)
    max_curve_with_bn = np.max(grads_with_bn, axis=0)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(min_curve_without_bn))

    plt.plot(x, min_curve_without_bn,
             label='Min Gradient without BN', color='blue')
    plt.plot(x, max_curve_without_bn, label='Max Gradient without BN',
             color='blue', linestyle='--')
    plt.fill_between(x, min_curve_without_bn,
                     max_curve_without_bn, color='blue', alpha=0.2)

    plt.plot(x, min_curve_with_bn, label='Min Gradient with BN', color='red')
    plt.plot(x, max_curve_with_bn, label='Max Gradient with BN',
             color='red', linestyle='--')
    plt.fill_between(x, min_curve_with_bn, max_curve_with_bn,
                     color='red', alpha=0.2)

    plt.xlabel('Training Steps')
    plt.ylabel('Gradient')
    plt.title(title)
    plt.legend()
    plt.grid(True)
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

    for lr in learning_rates:
        set_random_seeds(seed_value=2020, device=str(device))

        # Train model without BN
        model_without_bn = VGG_A()
        optimizer_without_bn = torch.optim.Adam(
            model_without_bn.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_model_path_without_bn = os.path.join(
            models_path, f'best_model_without_bn_{lr}_{timestamp}.pth')
        losses_without_bn, grads_without_bn = train(
            model_without_bn, optimizer_without_bn, criterion, train_loader, val_loader, epochs_n=epo, best_model_path=best_model_path_without_bn, lr=lr
        )
        all_losses_without_bn.extend(losses_without_bn)
        all_grads_without_bn.extend(grads_without_bn)

        # Train model with BN
        model_with_bn = VGG_A_BatchNorm()
        optimizer_with_bn = torch.optim.Adam(model_with_bn.parameters(), lr=lr)
        best_model_path_with_bn = os.path.join(
            models_path, f'best_model_with_bn_{lr}_{timestamp}.pth')
        losses_with_bn, grads_with_bn = train(
            model_with_bn, optimizer_with_bn, criterion, train_loader, val_loader, epochs_n=epo, best_model_path=best_model_path_with_bn, lr=lr
        )
        all_losses_with_bn.extend(losses_with_bn)
        all_grads_with_bn.extend(grads_with_bn)

    # Plot loss landscape
    plot_loss_landscape(all_losses_without_bn, all_losses_with_bn, 'Loss Landscape Comparison',
                        os.path.join(figures_path, 'loss_landscape_comparison.png'))

    # Plot gradient landscape
    plot_gradient_landscape(all_grads_without_bn, all_grads_with_bn, 'Gradient Landscape Comparison', os.path.join(
        figures_path, 'gradient_landscape_comparison.png'))
