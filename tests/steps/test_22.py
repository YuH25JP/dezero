import numpy as np

from steps.step22 import Variable


class TestNeg:
    def test_neg(self):
        x = Variable(np.array(2.0))
        y = -x
        expected = -2.0
        assert not isinstance(y, list)
        assert y.data == expected

    def test_neg_backward(self):
        x = Variable(np.array(2.0))
        y = -x
        assert not isinstance(y, list)
        y.backward()
        assert x.grad == -1.0


class TestSub:
    def test_sub(self):
        x = Variable(np.array(5.0))
        y = Variable(np.array(2.0))
        result = x - y
        expected = 3.0
        assert not isinstance(result, list)
        assert result.data == expected

    def test_sub_backward(self):
        x = Variable(np.array(5.0))
        y = Variable(np.array(2.0))
        result = x - y
        assert not isinstance(result, list)

        result.backward()
        assert x.grad == 1.0
        assert y.grad == -1.0

    def test_sub_variable_and_ndarray(self):
        x = Variable(np.array(5.0))
        y = np.array(2.0)
        result = x - y
        expected = 3.0
        assert not isinstance(result, list)
        assert result.data == expected

    def test_sub_variable_and_int(self):
        x = Variable(np.array(5.0))
        y = 2
        result = x - y
        expected = 3.0
        assert not isinstance(result, list)
        assert result.data == expected

    def test_sub_int_and_variable(self):
        x = Variable(np.array(2.0))
        y = 3 - x
        assert not isinstance(y, list)
        assert y.data == 1.0


class TestDiv:
    def test_div(self):
        x = Variable(np.array(6.0))
        y = Variable(np.array(2.0))
        result = x / y
        expected = 3.0
        assert not isinstance(result, list)
        assert result.data == expected

    def test_div_backward(self):
        x = Variable(np.array(6.0))
        y = Variable(np.array(2.0))
        result = x / y
        assert not isinstance(result, list)

        result.backward()
        assert x.grad == 1.0 / 2.0
        assert y.grad == -6.0 / (2.0) ** 2

    def test_div_int_and_variable(self):
        x = Variable(np.array(2.0))
        y = 6
        result = y / x
        assert not isinstance(result, list)
        assert result.data == 3.0


class TestPow:
    def test_pow(self):
        x = Variable(np.array(2.0))
        result = x**2
        expected = 4.0
        assert not isinstance(result, list)
        assert result.data == expected

    def test_pow_backward(self):
        x = Variable(np.array(2.0))
        result = x**2
        assert not isinstance(result, list)

        result.backward()
        assert x.grad == 2.0 * 2.0
