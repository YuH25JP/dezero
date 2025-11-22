import numpy as np

from steps.step20 import Variable, mul


class TestMultiply:
    def test_multiply(self):
        a = Variable(np.array(2.0))
        b = Variable(np.array(3.0))
        y = mul(a, b)
        assert not isinstance(y, list)

        y.backward()
        assert y.data == np.array(6.0)
        assert a.grad == np.float64(3.0)
        assert b.grad == np.float64(2.0)

    def test_multiply_overloaded(self):
        a = Variable(np.array(2.0))
        b = Variable(np.array(3.0))
        y = a * b
        assert not isinstance(y, list)

        y.backward()
        assert y.data == np.array(6.0)
        assert a.grad == np.float64(3.0)
        assert b.grad == np.float64(2.0)


class TestAdd:
    def test_add_overloaded(self):
        a = Variable(np.array(2.0))
        b = Variable(np.array(3.0))
        y = a + b
        assert not isinstance(y, list)

        y.backward()
        assert y.data == np.array(5.0)
        assert a.grad == np.float64(1.0)
        assert b.grad == np.float64(1.0)
