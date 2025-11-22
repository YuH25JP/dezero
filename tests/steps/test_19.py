import numpy as np

from steps.step19 import Variable


class TestVariableProperties:
    def test_variable_name(self):
        x = Variable(np.array(2.0), name="x_in")
        assert x.name == "x_in"

    def test_variable_name_None(self):
        x = Variable(np.array(2.0))
        assert x.name is None

    def test_shape_1d(self):
        x = Variable(np.array(2.0))
        expected = np.array(2.0).shape
        assert x.shape == expected

    def test_shape_2d(self):
        x = Variable(np.array([2.0, 4.0]))
        expected = np.array([2.0, 4.0]).shape
        assert x.shape == expected

    def test_ndim_1d(self):
        v = np.array([2.0])
        x = Variable(v)
        expected = v.ndim
        assert x.ndim == expected

    def test_ndim_2d(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = Variable(v)
        expected = v.ndim
        assert x.ndim == expected

    def test_size_1d(self):
        v = np.array([2.0])
        x = Variable(v)
        expected = v.size
        assert x.size == expected

    def test_size_2d(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = Variable(v)
        expected = v.size
        assert x.size == expected

    def test_dtype_1d(self):
        v = np.array([2.0])
        x = Variable(v)
        expected = v.dtype
        assert x.dtype == expected

    def test_dtype_2d(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = Variable(v)
        expected = v.dtype
        assert x.dtype == expected


class TestVariableLength:
    def test_len_1d(self):
        v = np.array([2.0])
        x = Variable(v)
        expected = 1
        assert len(x) == expected

    def test_len_2d(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = Variable(v)
        expected = 2
        assert len(x) == expected
