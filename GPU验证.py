import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# --- 1. 配置和超参数 ---

# 检查并设置设备 (GPU或CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"检测到 CUDA! 将使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("未检测到 CUDA. 将使用 CPU. (GPU测试将无法进行)")

# 定义超参数
NUM_EPOCHS = 10         # 训练周期数，可以增加此值以延长测试时间
BATCH_SIZE = 256        # 批处理大小，较大的批次可以更好地利用GPU并行能力
LEARNING_RATE = 0.001   # 学习率

# --- 2. 准备数据 ---

print("\n正在准备 CIFAR-10 数据集...")
# 定义数据预处理
# 对训练集进行数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 对测试集只进行标准化
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 下载并加载数据集
# num_workers > 0 可以使用多进程加载数据，加快数据准备速度
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("数据集准备完毕.")

# --- 3. 定义卷积神经网络模型 ---

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.05)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.fc_layer(x)
        return x

# --- 4. 初始化模型、损失函数和优化器 ---

# 实例化模型并将其移动到GPU
model = SimpleCNN().to(device)

# 确认模型所在的设备
print(f"模型已成功加载到: {next(model.parameters()).device}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. 训练模型 ---

print("\n" + "="*30)
print("     即将开始训练，请打开任务管理器或nvidia-smi监控GPU状态！     ")
print("="*30 + "\n")
time.sleep(3) # 等待3秒，方便您准备监控工具

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train() # 设置为训练模式
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # 将数据和标签移动到GPU
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 每100个批次打印一次信息
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# --- 6. 训练结束 ---
end_time = time.time()
total_time = end_time - start_time
print("\n训练完成!")
print(f"总训练耗时: {total_time:.2f} 秒")


# --- 7. 在测试集上评估模型 ---
model.eval() # 设置为评估模式
with torch.no_grad(): # 在评估阶段不计算梯度
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'\n模型在10000张测试图像上的准确率: {100 * correct / total:.2f} %')