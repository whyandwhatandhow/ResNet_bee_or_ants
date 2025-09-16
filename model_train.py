import copy
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model import ResNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_val_data_process(data_root: str = './hymenoptera_data'):
    """创建 ImageFolder 数据加载器，假设目录结构为
    hymenoptera_data/
      train/
        ants/
        bees/
      val/
        ants/
        bees/
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"未找到数据目录: {data_root}")

    input_size = 224
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=train_transform)
    val_set = datasets.ImageFolder(root=os.path.join(data_root, 'val'), transform=val_transform)

    # 增加batch_size以提高GPU利用率，设置pin_memory=True加速数据传输
    train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(dataset=val_set, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, train_set.classes


def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 打印设备信息
    if torch.cuda.is_available():
        print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("使用CPU训练")

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    best_most_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()

    for epoch in range(epochs):
        model.train()
        print("Epoch={}/{}".format(epoch + 1, epochs))
        print("-" * 10)
        train_loss = 0.0
        val_loss = 0.0

        train_acc = 0
        val_acc = 0

        train_num = 0
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_loader):
            model.train()
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            outputs = model(b_x)
            loss = criterion(outputs, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            preds = torch.argmax(outputs, 1)
            train_acc += torch.sum(preds == b_y.data).item()
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_loader):
            model.eval()
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            with torch.no_grad():
                outputs = model(b_x)
                loss = criterion(outputs, b_y)
                pred = torch.argmax(outputs, dim=1)

                val_loss += loss.item() * b_x.size(0)
                val_acc += torch.sum(pred == b_y.data).item()
                val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_acc_all.append(train_acc / train_num)
        val_acc_all.append(val_acc / val_num)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / train_num:.4f}, ACC: {train_acc / train_num:.4f}")
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss / val_num:.4f}, Val ACC: {val_acc / val_num:.4f}")

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_most_wts = copy.deepcopy(model.state_dict())
            torch.save(best_most_wts, "best_model_hy.pth")

    model.load_state_dict(best_most_wts)
    time_use = time.time() - since

    train_process = pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "train_loss": train_loss_all,
        "val_loss": val_loss_all,
        "train_acc": train_acc_all,
        "val_acc": val_acc_all,
    })
    train_process["time_use"] = time_use
    train_process["best_acc"] = best_acc

    return train_process


def matlib_acc_loss(train_process):
    epochs = train_process['epoch']
    train_loss = train_process['train_loss']
    val_loss = train_process['val_loss']
    train_acc = train_process['train_acc']
    val_acc = train_process['val_acc']

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)

    # 获取数据（ImageFolder）
    data_root = './hymenoptera_data'
    train_loader, val_loader, classes = train_val_data_process(data_root)
    num_classes = len(classes)
    print(f"检测到的类别: {classes} (num_classes={num_classes})")

    # 创建模型（3通道图像，类别为数据集类别数）
    model = ResNet(in_channels=3, num_classes=num_classes)

    # 训练模型
    print("开始训练...")
    process = train_model(model, train_loader, val_loader, 20)

    # 绘制训练曲线
    matlib_acc_loss(process)

    # 打印训练结果
    print("\n训练结果:")
    print(process)
    print("#"*20)
    print(f"\n最佳准确率: {process['best_acc'].iloc[0]:.4f}")
    print(f"训练时间: {process['time_use'].iloc[0]:.2f}秒")
