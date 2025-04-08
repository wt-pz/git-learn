import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# 数据加载预处理
def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    y = mnist.target.astype(np.int_)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# 卷积层实现
class ConvLayer:
    def __init__(self, in_ch, out_ch, kernel_size):
        self.W = np.random.randn(out_ch, in_ch, kernel_size, kernel_size) * np.sqrt(2.0 / (in_ch * kernel_size ** 2))
        self.b = np.zeros(out_ch)
        self.kernel_size = kernel_size

    def forward(self, x):
        batch, in_ch, h, w = x.shape
        out_h = h - self.kernel_size + 1
        out_w = w - self.kernel_size + 1
        self.x = x
        output = np.zeros((batch, self.W.shape[0], out_h, out_w))

        for b in range(batch):
            for oc in range(self.W.shape[0]):
                for ih in range(out_h):
                    for iw in range(out_w):
                        region = x[b, :, ih:ih + self.kernel_size, iw:iw + self.kernel_size]
                        output[b, oc, ih, iw] = np.sum(region * self.W[oc]) + self.b[oc]
        return output

    def backward(self, dout, lr=0.01):
        batch, out_ch, out_h, out_w = dout.shape
        dW = np.zeros_like(self.W)
        db = np.sum(dout, axis=(0, 2, 3))
        dx = np.zeros_like(self.x)

        for b in range(batch):
            for oc in range(out_ch):
                for ih in range(out_h):
                    for iw in range(out_w):
                        region = self.x[b, :, ih:ih + self.kernel_size, iw:iw + self.kernel_size]
                        dW[oc] += region * dout[b, oc, ih, iw]
                        dx[b, :, ih:ih + self.kernel_size, iw:iw + self.kernel_size] += self.W[oc] * dout[b, oc, ih, iw]

        self.W -= lr * dW / batch
        self.b -= lr * db / batch
        return dx


# ReLU激活函数
class ReLU:
    def forward(self, x):
        self.mask = x < 0
        return np.maximum(0, x)

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


# 最大池化层
class MaxPool2d:
    def __init__(self, size=2):
        self.size = size

    def forward(self, x):
        batch, ch, h, w = x.shape
        self.x = x
        out_h = h // self.size
        out_w = w // self.size
        self.arg_max = np.zeros((batch, ch, out_h, out_w), dtype=int)

        output = np.zeros((batch, ch, out_h, out_w))
        for b in range(batch):
            for c in range(ch):
                for i in range(out_h):
                    for j in range(out_w):
                        hs = i * self.size
                        ws = j * self.size
                        region = x[b, c, hs:hs + self.size, ws:ws + self.size]
                        output[b, c, i, j] = np.max(region)
                        self.arg_max[b, c, i, j] = np.unravel_index(region.argmax(), region.shape)
        return output

    def backward(self, dout):
        dx = np.zeros_like(self.x)
        batch, ch, out_h, out_w = dout.shape

        for b in range(batch):
            for c in range(ch):
                for i in range(out_h):
                    for j in range(out_w):
                        hs = i * self.size
                        ws = j * self.size
                        h_max, w_max = self.arg_max[b, c, i, j]
                        dx[b, c, hs + h_max, ws + w_max] = dout[b, c, i, j]
        return dx


# 全连接层
class FCLayer:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros(out_dim)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout, lr=0.01):
        dx = dout @ self.W.T
        dW = self.x.T @ dout
        db = np.sum(dout, axis=0)

        self.W -= lr * dW / len(dout)
        self.b -= lr * db / len(dout)
        return dx


# Softmax交叉熵损失
class SoftmaxCELoss:
    def forward(self, x, y):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        loss = -np.log(self.probs[np.arange(len(y)), y]).mean()
        return loss

    def backward(self, y):
        batch = len(y)
        grad = self.probs
        grad[np.arange(batch), y] -= 1
        return grad / batch


# 网络结构
class CNN:
    def __init__(self):
        self.layers = [
            ConvLayer(1, 8, 3),
            ReLU(),
            MaxPool2d(2),
            ConvLayer(8, 16, 3),
            ReLU(),
            MaxPool2d(2),
            FCLayer(16 * 5 * 5, 128),
            ReLU(),
            FCLayer(128, 10)
        ]
        self.loss = SoftmaxCELoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            if isinstance(layer, (ConvLayer, FCLayer)):
                grad = layer.backward(grad, lr)
            else:
                grad = layer.backward(grad)
        return grad


# 训练流程
X_train, X_test, y_train, y_test = load_data()
cnn = CNN()
batch_size = 64
epochs = 5

for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        # 前向传播
        x_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        output = cnn.forward(x_batch)

        # 计算损失
        loss = cnn.loss.forward(output, y_batch)

        # 反向传播
        grad = cnn.loss.backward(y_batch)
        cnn.backward(grad, lr=0.01)

    # 验证
    test_pred = cnn.forward(X_test).argmax(axis=1)
    acc = (test_pred == y_test).mean()
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Acc: {acc:.4f}")