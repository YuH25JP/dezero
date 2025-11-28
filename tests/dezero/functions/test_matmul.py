import numpy as np
from numpy.testing import assert_array_equal

import dezero.functions as F
from dezero import Variable


class TestMatmul:
    def test_forward(self):
        npx = np.array([[1, 2, 3], [4, 5, 6]])
        npW = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        x = Variable(npx)
        W = Variable(npW)
        y = F.matmul(x, W)
        assert isinstance(y, Variable)
        assert_array_equal(y.data, npx @ npW)

    # TODO: Write backward test for matmul
