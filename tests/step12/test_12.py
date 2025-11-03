import numpy as np
from steps.step12 import Variable, add


def test_add():
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)

    is_none = isinstance(y, list)
    expected = np.array(5)
    assert not is_none and y.data == expected
