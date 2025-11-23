import numpy as np

from dezero import Variable

x = Variable(np.array(2.0))
y = (x + 3) ** 2
y.backward()

print(y)
