import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time


# --- 将模型定义和函数定义放在顶层 ---
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


# --- 主执行逻辑必须放在这个 if 语句块内 ---
if __name__ == '__main__':
    # --- 1. 配置和超参数 ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"检测到 CUDA! 将使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("未检测到 CUDA. 将使用 CPU. (GPU测试将无法进行)")

    NUM_EPOCHS = 10
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001

    # --- 2. 准备数据 ---
    print("\n正在准备 CIFAR-10 数据集...")
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

    # 在Windows下使用多进程，num_workers建议不要设置过高，4是比较合理的值
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print("数据集准备完毕.")

    # --- 3. 初始化模型、损失函数和优化器 ---
    model = SimpleCNN().to(device)
    print(f"模型已成功加载到: {next(model.parameters()).device}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. 训练模型 ---
    print("\n" + "=" * 30)
    print("     即将开始训练，请打开任务管理器或nvidia-smi监控GPU状态！     ")
    print("=" * 30 + "\n")
    time.sleep(3)
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # --- 5. 训练结束与评估 ---
    end_time = time.time()
    total_time = end_time - start_time
    print("\n训练完成!")
    print(f"总训练耗时: {total_time:.2f} 秒")

    model.eval()
    with torch.no_grad():
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