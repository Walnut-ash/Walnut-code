import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),    # 将图像大小调整为64x64
    transforms.ToTensor(),           # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 自定义数据集类
class PseudoLabelDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, 0  # 将所有图像视为同一类别，标签为0

# 加载数据集
train_dataset = PseudoLabelDataset(root='./CarDD/CarDD_COCO/train2017', transform=transform)
test_dataset = PseudoLabelDataset(root='./CarDD/CarDD_COCO/test2017', transform=transform)
val_dataset = PseudoLabelDataset(root='./CarDD/CarDD_COCO/val2017', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


# 定义模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)   # 注意：64x16x16是根据输入图像大小调整的
        self.fc2 = nn.Linear(128, 1)             # 输出层的大小为1，因为只有一个类别
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = nn.functional.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


# 实例化模型、定义损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()  # 由于只有一个类别，使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), running_loss / 100))
            running_loss = 0.0


    # 验证模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('验证集准确率: {:.2f}%'.format(100 * correct / total))


# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('测试集准确率: {:.2f}%'.format(100 * correct / total))
