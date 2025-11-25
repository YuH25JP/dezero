import numpy as np

from dezero import Variable
from dezero.utils import plot_dot_graph

x = Variable(np.array(2.0), name="x")
y = x**2
assert isinstance(y, Variable)
y.name = "y"
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

assert isinstance(gx, Variable)
gx.name = "gx"
z = gx**3 + y
assert isinstance(z, Variable)
z.name = "z"
z.backward()

plot_dot_graph(z, to_file="double_backprop.png")
print(x.grad)
