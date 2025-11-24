import numpy as np

from dezero import Variable


def f(x):
    y = x**4 - 2 * x**2
    y = x**4 - 2 * x**2
    return y


x = Variable(np.array(2.0))
iters = 10

for i in range(10):
    print(i, x)

    y = f(x)
    x.cleargrad()
    assert isinstance(y, Variable)
    y.backward(create_graph=True)
    gx = x.grad
    x.cleargrad()
    assert isinstance(gx, Variable)
    gx.backward()
    gx2 = x.grad
    assert isinstance(gx2, Variable)

    x.data -= gx.data / gx2.data
