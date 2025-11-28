import numpy as np
from numpy.testing import assert_array_equal

from dezero import Variable


class TestBroadcastAdd:
    def test_broadcast_add_forward(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 + x1
        assert isinstance(y, Variable)
        assert_array_equal(y.data, np.array([11, 12, 13]))

    def test_broadcast_add_backward(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 + x1
        assert isinstance(y, Variable)

        y.backward()
        assert x0.grad is not None
        assert x1.grad is not None

        assert_array_equal(x0.grad.data, np.array([1, 1, 1]))
        assert_array_equal(x1.grad.data, np.array([3]))


class TestBroadcastSub:
    def test_broadcast_sub_forward(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 - x1
        assert isinstance(y, Variable)
        assert_array_equal(y.data, np.array([-9, -8, -7]))

    def test_broadcast_sub_backward(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 - x1
        assert isinstance(y, Variable)

        y.backward()
        assert x0.grad is not None
        assert x1.grad is not None

        assert_array_equal(x0.grad.data, np.array([1, 1, 1]))
        assert_array_equal(x1.grad.data, np.array([-3]))


class TestBroadcastMul:
    def test_broadcast_mul_forward(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 * x1
        assert isinstance(y, Variable)
        assert_array_equal(y.data, np.array([10, 20, 30]))

    def test_broadcast_mul_backward(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 * x1
        assert isinstance(y, Variable)

        y.backward()
        assert x0.grad is not None
        assert x1.grad is not None

        assert_array_equal(x0.grad.data, np.array([10, 10, 10]))
        assert_array_equal(x1.grad.data, np.array([6]))


class TestBroadcastDiv:
    def test_broadcast_div_forward(self):
        x0 = Variable(np.array([10, 20, 30]))
        x1 = Variable(np.array([10]))
        y = x0 / x1
        assert isinstance(y, Variable)
        assert_array_equal(y.data, np.array([1, 2, 3]))

    def test_broadcast_div_backward(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 / x1
        assert isinstance(y, Variable)

        y.backward()
        assert x0.grad is not None
        assert x1.grad is not None

        assert_array_equal(x0.grad.data, np.array([0.1, 0.1, 0.1]))
        assert_array_equal(x1.grad.data, np.array([-0.06]))
