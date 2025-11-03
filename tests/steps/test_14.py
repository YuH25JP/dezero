import numpy as np
from steps.step14 import Variable, square, add


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


class TestSameVariable:
    def test_forward(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        y_is_list = isinstance(y, list)

        if not y_is_list:
            y.backward()

        assert not y_is_list and x.grad == np.array(2.0)


class TestRepeatedUseOfVariable:
    def test_repeated_use(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        y_is_list_first = isinstance(y, list)

        if not y_is_list_first:
            y.backward()

        correct_fist = not y_is_list_first and x.grad == np.array(2.0)

        x.cleargrad()
        y = add(add(x, x), x)
        y_is_list_second = isinstance(y, list)

        if not y_is_list_second:
            y.backward()

        correct_second = not y_is_list_second and x.grad == np.array(3.0)

        assert correct_fist and correct_second
