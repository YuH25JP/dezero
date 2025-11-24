import numpy as np

from dezero import Variable


class TestNeg:
    def test_forward(self):
        x = Variable(np.array(2.0))
        z = -x
        assert not isinstance(z, list)
        assert z.data == -2.0

    def test_backward(self):
        x = Variable(np.array(2.0))
        z = -x
        assert not isinstance(z, list)

        z.backward()
        assert isinstance(x.grad, Variable)
        assert x.grad.data == -1.0
