# Batch-Normalization


### 一、项目概述：

Batch Normalization (BN) 是一种广泛应用于深度神经网络 (DNNs) 的技术，它能加速训练过程并提高训练的稳定性。本项目旨在测试 BN 在训练过程中的有效性，并探究其对优化过程的帮助。


### 二、数据集介绍

本实验使用的 [CIFAR - 10 ](https://www.cs.toronto.edu/~kriz/cifar.html) 数据集是一个广泛用于图像分类研究的公开数据集。
- **数据内容**：60,000张32×32像素的彩色图像
- **图像类别**：10个类别（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）
- **数据划分**：
  - 训练集：50,000张（5,000张/类）
  - 测试集：10,000张（1,000张/类）
- **图像通道**：RGB三通道

### 三、文件结构

```
VGG_BatchNorm/
├── data/
│   ├── __init__.py
│   └── loaders.py         # 负责下载和加载 CIFAR - 10 数据集
├── models/
│   ├── __init__.py
│   └── vgg.py             # 定义了不同版本的 VGG 模型
├── utils/
│   ├── __init__.py
│   └── nn.py              # 包含神经网络相关的工具函数
├── VGG_Loss_Landscape.py  # 主程序文件，负责整个实验流程，包括数据加载、模型训练、评估以及损失景观和梯度景观的绘制。
```

### 四、实验步骤


#### 1. 环境准备：

使用以下命令安装必要的 Python 库:
```
pip install torch torchvision matplotlib tqdm
```

#### 2. 数据加载：

运行 `data/loaders.py` 下载 CIFAR - 10 数据集，并加载数据。

```
python data/loaders.py
```

#### 3. 模型定义：

**VGG - A 模型：**在 `models/vgg.py` 中定义了 `VGG_A` 类，该类实现了基本的 VGG - A 网络结构。
**VGG - A with BN 模型：**同样在 `models/vgg.py` 中定义了 `VGG_A_BatchNorm` 类，该类在 `VGG_A` 的基础上添加了 Batch Normalization 层。

#### 4. 模型训练与可视化评估：

运行 `VGG_Loss_Landscape.py` 脚本进行模型训练和评估。其中 `train` 函数用于完成整个训练过程，记录每个步骤的损失值和梯度，并在每个 epoch 结束时计算训练集和验证集的准确率。`get_accuracy` 函数用于计算模型在数据集上的准确率。`plot_loss_landscape` 函数用于绘制损失景观图，对比使用 BN 和不使用 BN 的模型在训练过程中的损失变化范围。`plot_gradient_landscape` 函数用于绘制梯度景观图，对比使用 BN 和不使用 BN 的模型在训练过程中的梯度变化范围。在 `__main__` 函数中，设置了多个学习率 [1e-3, 2e-3, 1e-4, 5e-4]，分别对使用 BN 和不使用 BN 的模型进行训练。

```
python VGG_Loss_Landscape.py
```



### 五、模型权重下载
模型和可视化结果保存在我的网盘: 