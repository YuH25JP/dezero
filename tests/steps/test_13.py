import numpy as np
from steps.step13 import Variable, square


class TestExp:
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        assert not isinstance(y, list) and y.data == expected

    def test_backward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        y_is_list = isinstance(y, list)

        if not y_is_list:
            y.backward()
            gx_correct = x.grad == np.array(4.0)
        else:
            gx_correct = False

        assert not y_is_list and gx_correct
