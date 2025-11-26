import numpy as np
from numpy.testing import assert_array_equal

from dezero import Variable
from dezero.functions import transpose


class TestTranspose:
    def test_forward(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = transpose(x)
        assert isinstance(y, Variable)
        assert_array_equal(y.data, np.array([[1, 4], [2, 5], [3, 6]]))

    def test_backward(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = transpose(x)
        assert isinstance(y, Variable)
        y.backward()
        assert isinstance(x.grad, Variable)
        assert_array_equal(x.grad.data, np.array([[1, 1, 1], [1, 1, 1]]))

    def test_forward_as_method(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.transpose()
        assert isinstance(y, Variable)
        assert_array_equal(y.data, np.array([[1, 4], [2, 5], [3, 6]]))

    def test_forward_as_property(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.T
        assert isinstance(y, Variable)
        assert_array_equal(y.data, np.array([[1, 4], [2, 5], [3, 6]]))
