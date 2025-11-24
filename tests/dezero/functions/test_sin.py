import numpy as np
from pytest import approx

from dezero import Variable
from dezero.functions import sin


class TestSin:
    def test_forward(self):
        x = Variable(np.array(np.pi / 4))
        y = sin(x)
        assert isinstance(y, Variable)
        assert y.data == np.sin(np.pi / 4)

    def test_backward(self):
        x = Variable(np.array(np.pi / 4))
        y = sin(x)
        assert isinstance(y, Variable)
        y.backward()
        assert x.grad is not None
        assert x.grad.data == np.cos(np.pi / 4)


class TestSinHigherDerivatives:
    def test_sin_fourth(self):
        x = Variable(np.array(1.0))
        y = sin(x)
        assert isinstance(y, Variable)
        y.backward(create_graph=True)

        gs = []
        for i in range(3):
            gx = x.grad
            x.cleargrad()
            assert isinstance(gx, Variable)
            gx.backward(create_graph=True)
            gs.append(x.grad)

        assert gs[0].data == approx(-0.8414709848)
        assert gs[1].data == approx(-0.5403023058)
        assert gs[2].data == approx(0.8414709848)
