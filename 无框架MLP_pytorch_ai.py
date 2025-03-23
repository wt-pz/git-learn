# mlp_fashion_mnist_torch.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 加载Fashion-MNIST数据集
fashion_mnist = fetch_openml('Fashion-MNIST', version=1)
X = fashion_mnist.data.astype('float32') / 255.0
y = fashion_mnist.target.astype('int')
y = y.values
# 将标签转换为one-hot编码
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1)).astype('float32')  # Ensure float32

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
# 转换为PyTorch张量
X_train = torch.tensor(X_train.values if hasattr(X_train, 'values') else np.array(X_train), dtype=torch.float32)
X_test = torch.tensor(X_test.values if hasattr(X_test, 'values') else np.array(X_test), dtype=torch.float32)
y_train = torch.tensor(y_train.values if hasattr(y_train, 'values') else np.array(y_train), dtype=torch.float32)
y_test = torch.tensor(y_test.values if hasattr(y_test, 'values') else np.array(y_test), dtype=torch.float32)

# 创建DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义MLP模型
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = torch.randn(input_size, hidden_size, requires_grad=True, dtype=torch.float32) * 0.01
        self.b1 = torch.zeros(1, hidden_size, requires_grad=True, dtype=torch.float32)
        self.W2 = torch.randn(hidden_size, output_size, requires_grad=True, dtype=torch.float32) * 0.01
        self.b2 = torch.zeros(1, output_size, requires_grad=True, dtype=torch.float32)

    def relu(self, x):
        return torch.max(torch.zeros_like(x), x)

    def softmax(self, x):
        exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
        return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -torch.log(y_pred[range(m), y_true.argmax(dim=1)])
        loss = torch.sum(log_likelihood) / m
        return loss

    def backward(self, X, y_true, learning_rate):
        m = y_true.shape[0]

        # 输出层误差
        dz2 = self.a2 - y_true
        dW2 = torch.matmul(self.a1.T, dz2) / m
        db2 = torch.sum(dz2, dim=0, keepdim=True) / m

        # 隐藏层误差
        dz1 = torch.matmul(dz2, self.W2.T) * (self.a1 > 0).float()
        dW1 = torch.matmul(X.T, dz1) / m
        db1 = torch.sum(dz1, dim=0, keepdim=True) / m

        # 更新参数
        with torch.no_grad():
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

    def train(self, train_loader, epochs, learning_rate):
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                y_pred = self.forward(X_batch)
                loss = self.cross_entropy_loss(y_pred, y_batch)
                self.backward(X_batch, y_batch, learning_rate)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

    def predict(self, X):
        y_pred = self.forward(X)
        return torch.argmax(y_pred, dim=1)

# 初始化模型
input_size = X_train.shape[1]
hidden_size = 128
output_size = y_train.shape[1]
mlp = MLP(input_size, hidden_size, output_size)

# 训练模型
mlp.train(train_loader, epochs=100, learning_rate=0.01)

# 测试模型
y_pred = mlp.predict(X_test)
accuracy = torch.mean((y_pred == torch.argmax(y_test, dim=1)).float()).item()
print(f'Test Accuracy: {accuracy * 100:.2f}%')