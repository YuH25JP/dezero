import numpy as np
from pytest import approx

from dezero import Variable
from dezero.functions import tanh


class TestTanh:
    def test_forward(self):
        x = Variable(np.array(np.pi / 4))
        y = tanh(x)
        assert isinstance(y, Variable)
        assert y.data == approx(np.tanh(np.pi / 4))

    def test_backward(self):
        x = Variable(np.array(np.pi / 4))
        y = tanh(x)
        assert isinstance(y, Variable)
        y.backward()
        assert x.grad is not None
        assert x.grad.data == approx(1 - np.tanh(np.pi / 4) ** 2)
