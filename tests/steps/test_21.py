import numpy as np

from steps.step21 import Variable, as_variable


class TestAsvariable:
    def test_ndarray_as_variable(self):
        x = np.array(2.0)
        y = as_variable(x)
        expected = Variable(np.array(2.0))
        assert isinstance(y, Variable)
        assert y.data == expected.data

    def test_variable_as_variable(self):
        x = Variable(np.array(2.0))
        y = as_variable(x)
        assert isinstance(y, Variable)


class TestNdarrayMixed:
    def test_variable_and_ndarray(self):
        """`variable` + `ndarray`"""
        x = Variable(np.array(2.0))
        y = x + np.array(3.0)
        expected = 5.0
        assert not isinstance(y, list)
        assert y.data == expected

    def test_ndarray_and_variable(self):
        """`ndarray` + `variable`

        This test can be passed thanks to `__array_priority__` property in Variable
        """
        x = Variable(np.array([2.0]))
        y = np.array([3.0]) + x
        expected = 5.0
        assert y.data == expected


class TestPrimaryTypeMixed:
    def test_add_variable_and_int(self):
        x = Variable(np.array(2.0))
        y = x + 2
        expected = 4.0
        assert not isinstance(y, list)
        assert y.data == expected

    def test_add_int_and_variable(self):
        x = Variable(np.array(2.0))
        y = 2 + x
        expected = 4.0
        assert not isinstance(y, list)
        assert y.data == expected

    def test_add_variable_and_float(self):
        x = Variable(np.array(2.0))
        y = x + 2.0
        expected = 4.0
        assert not isinstance(y, list)
        assert y.data == expected

    def test_add_float_and_variable(self):
        x = Variable(np.array(2.0))
        y = 2.0 + x
        expected = 4.0
        assert not isinstance(y, list)
        assert y.data == expected

    def test_mul_variable_and_int(self):
        x = Variable(np.array(2.0))
        y = x * 2
        expected = 4.0
        assert not isinstance(y, list)
        assert y.data == expected

    def test_mul_variable_and_float(self):
        x = Variable(np.array(2.0))
        y = x * 2.0
        expected = 4.0
        assert not isinstance(y, list)
        assert y.data == expected

    def test_mul_int_and_variable(self):
        x = Variable(np.array(2.0))
        y = 2 * x
        expected = 4.0
        assert not isinstance(y, list)
        assert y.data == expected
