import numpy as np
from numpy.testing import assert_array_equal

from dezero import Variable
from dezero.functions import sum


class TestSum:
    def test_forward(self):
        v = Variable(np.array([1, 2, 3, 4, 5, 6]))
        y = sum(v)
        assert isinstance(y, Variable)
        assert y.data == 21

    def test_backward(self):
        v = Variable(np.array([1, 2, 3, 4, 5, 6]))
        y = sum(v)
        assert isinstance(y, Variable)
        y.backward()
        assert v.grad is not None
        assert_array_equal(v.grad.data, np.array([1, 1, 1, 1, 1, 1]))
