import numpy as np

from dezero import Variable


class TestPow:
    def test_forward(self):
        x = Variable(np.array(3.0))
        y = 4
        z = x**y
        assert not isinstance(z, list)
        assert z.data == 81.0

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = 4
        z = x**y
        assert not isinstance(z, list)

        z.backward()
        assert isinstance(x.grad, Variable)
        assert x.grad.data == 27.0 * 4
