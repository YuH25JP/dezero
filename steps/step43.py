import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
from dezero import Variable

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))


def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


lr = 0.2
iters = 10000

loss_hist = []

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    loss_hist.append(loss.data)
    if i % 1000 == 0:
        print(i, loss)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax1.plot(loss_hist)
ax1.set_xlabel("iter")
ax1.set_ylabel("loss")

ax2 = fig.add_subplot(122)
xx = np.linspace(0, 1, 100).reshape((100, 1))
y_model = predict(xx).data
ax2.scatter(x, y, label="data")
ax2.plot(xx, y_model, color="red", label="predicted")
ax2.legend()
ax2.set_xlabel("x")
ax2.set_ylabel("y")

plt.show()
