from data.loaders import get_cifar_loader
from models.vgg import VGG_A_BatchNorm  # you need to implement this network
from models.vgg import VGG_A
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
    # --------------------
    # Add code as needed
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

    # --------------------

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
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
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
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            # --------------------
            # Add your code
            #
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
            # --------------------

        losses_list.append(loss_list)
        grads.append(grad)
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 5))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve[:epoch+1])
        axes[0].set_title('Training Loss')

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        # --------------------
        # Add code as needed
        #
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
            figures_path, f'training_progress_epoch{epoch+1}.png'), dpi=300)
        plt.close()  # Prevent memory leak

        print(f'Epoch {epoch+1}/{epochs_n}')
        print(f'Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')
        print(
            f'Best Val Accuracy: {max_val_accuracy:.4f} at epoch {max_val_accuracy_epoch+1}')
        # --------------------

    return losses_list, grads, learning_curve, train_accuracy_curve, val_accuracy_curve


# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(losses):
    min_curve = np.min(losses, axis=0)
    max_curve = np.max(losses, axis=0)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(min_curve))
    plt.plot(x, min_curve, label='Min Loss')
    plt.plot(x, max_curve, label='Max Loss')
    plt.fill_between(x, min_curve, max_curve, alpha=0.2)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Landscape')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_path, 'loss_landscape.png'), dpi=300)
    plt.close()


# 保存样本图像函数
def save_sample_images(loader, count=16, path=None):
    """保存样本图像及其标签"""
    if path is None:
        path = os.path.join(figures_path, 'sample_images.png')

    X, y = next(iter(loader))
    plt.figure(figsize=(10, 10))

    for i in range(min(count, X.size(0))):
        plt.subplot(4, 4, i+1)
        plt.imshow(X[i].permute(1, 2, 0))  # 调整通道顺序
        plt.title(f'Label: {y[i]}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# 保存预测结果可视化函数
def save_prediction_examples(model, loader, class_names, count=16, path=None):
    """保存模型预测结果可视化"""
    if path is None:
        path = os.path.join(figures_path, 'prediction_examples.png')

    model.eval()
    X, y = next(iter(loader))
    X = X.to(device)

    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)

    plt.figure(figsize=(10, 10))

    for i in range(min(count, X.size(0))):
        plt.subplot(4, 4, i+1)
        plt.imshow(X[i].cpu().permute(1, 2, 0))  # 调整通道顺序
        true_label = class_names[y[i]]
        pred_label = class_names[predicted[i]]
        plt.title(f'True: {true_label}\nPred: {pred_label}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=300)
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

    set_random_seeds(seed_value=2020, device=str(device))
    model = VGG_A()
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 定义带时间戳的模型保存路径
    best_model_path = os.path.join(models_path, f'best_model_{timestamp}.pth')
    final_model_path = os.path.join(
        models_path, f'final_model_{timestamp}.pth')

    # 执行训练并获取额外返回值
    losses, grads, learning_curve, train_accuracy_curve, val_accuracy_curve = train(
        model, optimizer, criterion, train_loader, val_loader, epochs_n=epo, best_model_path=best_model_path)

    # 保存最终模型
    torch.save(model.state_dict(), final_model_path)

    # 保存loss和gradient数据（带时间戳）
    np.savetxt(os.path.join(
        figures_path, f'losses_{timestamp}.txt'), losses, fmt='%s')
    np.savetxt(os.path.join(
        figures_path, f'grads_{timestamp}.txt'), grads, fmt='%s')

    # 保存训练历史为CSV
    history_df = pd.DataFrame({
        'epoch': range(epo),
        'train_loss': learning_curve,
        'train_accuracy': train_accuracy_curve,
        'val_accuracy': val_accuracy_curve
    })
    history_df.to_csv(os.path.join(
        figures_path, f'training_history_{timestamp}.csv'), index=False)

    # Plot loss landscape
    plot_loss_landscape(losses)

    # 假设class_names是CIFAR-10的类别名称（根据实际数据调整）
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 保存样本图像和预测示例
    save_sample_images(train_loader)
    save_prediction_examples(model, val_loader, class_names)

    print(f"所有结果已保存至: {figures_path} 和 {models_path}")
