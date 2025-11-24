import numpy as np

from dezero import Variable


class TestAdd:
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = x + y
        assert not isinstance(z, list)
        assert z.data == 5.0

    def test_backward(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = x + y
        assert not isinstance(z, list)

        z.backward()
        assert isinstance(x.grad, Variable)
        assert isinstance(y.grad, Variable)
        assert x.grad.data == 1.0
        assert y.grad.data == 1.0
