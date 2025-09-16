import torch
import torch.nn as nn
from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_1x1conv=False):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, X):
        # 保存原始输入用于残差连接
        residual = X

        # 主路径
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        if self.conv3:
            residual = self.conv3(residual)  # 调整原始输入的维度

        return self.relu(out + residual)  # 正确的残差连接


class ResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(ResNet, self).__init__()
        # 为MNIST 28x28输入调整的初始层
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.b2 = nn.Sequential(
            Residual(in_channels=64, out_channels=64),
            Residual(in_channels=64, out_channels=64),
        )
        self.b3 = nn.Sequential(
            Residual(in_channels=64, out_channels=128, use_1x1conv=True, stride=2),
            Residual(in_channels=128, out_channels=128),
        )
        self.b4 = nn.Sequential(
            Residual(in_channels=128, out_channels=256, use_1x1conv=True, stride=2),
            Residual(in_channels=256, out_channels=256),
        )
        self.b5 = nn.Sequential(
            Residual(in_channels=256, out_channels=512, use_1x1conv=True, stride=2),
            Residual(in_channels=512, out_channels=512),
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, X):
        X = self.b1(X)
        X = self.b2(X)
        X = self.b3(X)
        X = self.b4(X)
        X = self.b5(X)
        X = self.b6(X)
        return X


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(in_channels=3, num_classes=2).to(device)
    if torch.cuda.is_available():
        print("当前使用的设备: CUDA (GPU)")
        print("CUDA 设备数量:", torch.cuda.device_count())
        print("当前CUDA设备索引:", torch.cuda.current_device())
        print("CUDA设备名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("当前使用的设备: CPU")
    print(summary(model, input_size=(3, 224, 224)))
    