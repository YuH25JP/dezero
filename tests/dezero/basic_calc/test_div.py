import numpy as np

from dezero import Variable


class TestDiv:
    def test_forward(self):
        x = Variable(np.array(12.0))
        y = Variable(np.array(4.0))
        z = x / y
        assert not isinstance(z, list)
        assert z.data == 3.0

    def test_backward(self):
        x = Variable(np.array(12.0))
        y = Variable(np.array(4.0))
        z = x / y
        assert not isinstance(z, list)

        z.backward()
        assert isinstance(x.grad, Variable)
        assert isinstance(y.grad, Variable)
        assert x.grad.data == 1.0 / 4.0
        assert y.grad.data == -12.0 / 16.0
