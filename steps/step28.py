import matplotlib.pyplot as plt
import numpy as np

from dezero import Variable


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

# Implement gradient descent
lr = 0.001
iters = 10000

hist = []

for i in range(iters):
    # print(x0, x1)

    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    assert not isinstance(y, list)
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

    hist.append((x0.data.tolist(), x1.data.tolist()))


x = np.linspace(-2, 2, 1000)
y = np.linspace(-1, 3, 1000)
xx, yy = np.meshgrid(x, y)
ros_xy = rosenbrock(xx, yy)

fig, ax = plt.subplots()
ax.contour(xx, yy, ros_xy, levels=np.logspace(0, 3, 10))
ax.scatter([h[0] for h in hist], [h[1] for h in hist])
ax.plot(1.0, 1.0, marker="*", color="red", markersize=20)
ax.set_xlabel("x0")
ax.set_ylabel("x1")
plt.show()
