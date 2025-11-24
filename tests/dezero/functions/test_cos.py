import numpy as np
from pytest import approx

from dezero import Variable
from dezero.functions import cos


class TestSin:
    def test_forward(self):
        x = Variable(np.array(np.pi / 4))
        y = cos(x)
        assert isinstance(y, Variable)
        assert y.data == approx(np.sin(np.pi / 4))

    def test_backward(self):
        x = Variable(np.array(np.pi / 4))
        y = cos(x)
        assert isinstance(y, Variable)
        y.backward()
        assert x.grad is not None
        # assert x.grad.shape == x.shape
        assert x.grad.data == approx(-np.sin(np.pi / 4))
