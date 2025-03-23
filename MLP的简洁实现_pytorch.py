# mlp_fashion_mnist_torch_concise.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 加载Fashion-MNIST数据集
fashion_mnist = fetch_openml('Fashion-MNIST', version=1)
X = fashion_mnist.data.astype('float32') / 255.0
y = fashion_mnist.target.astype('int')

# 将标签转换为one-hot编码
encoder = OneHotEncoder(sparse=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 创建DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# 初始化模型
input_size = X_train.shape[1]
hidden_size = 128
output_size = y_train.shape[1]
mlp = MLP(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.01)

# 训练模型
for epoch in range(100):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = mlp(X_batch)
        loss = criterion(y_pred, y_batch.argmax(dim=1))
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 测试模型
mlp.eval()
with torch.no_grad():
    y_pred = mlp(X_test)
    accuracy = (y_pred.argmax(dim=1) == y_test.argmax(dim=1)).float().mean().item()
    print(f'Test Accuracy: {accuracy * 100:.2f}%')