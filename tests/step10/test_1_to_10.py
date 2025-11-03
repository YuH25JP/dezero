import numpy as np

from steps.step09 import Variable, square, exp, numerical_diff


class TestSquare:
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        assert y.data == expected

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        assert x.grad == expected

    def test_gradient_check(self):
        """Test by comparing two different gradient values by autodiff and numerical algorithms."""
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()

        num_grad = numerical_diff(square, x)

        flg = False
        if x.grad is not None:
            flg = np.allclose(x.grad, num_grad)

        assert flg


class TestExp:
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = exp(x)
        expected = np.array(np.exp(2.0))
        assert y.data == expected

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = exp(x)
        y.backward()
        expected = np.array(np.exp(3.0))
        assert x.grad == expected

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = exp(x)
        y.backward()

        num_grad = numerical_diff(exp, x)

        flg = False
        if x.grad is not None:
            flg = np.allclose(x.grad, num_grad)

        assert flg
