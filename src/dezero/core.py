from __future__ import annotations  # lazy evaluation

import contextlib
import weakref

import numpy as np

import dezero


class Config:
    enable_backdrop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    """Helper function for disabling `enable_backdrop` in contextmanager.

    Examples
    --------
    >>> with no_grad():
            x = Variable(np.array(2.0))
            y = square(x)
    """
    return using_config("enable_backdrop", False)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)

    return x


def as_variable(obj):
    """Converts ndarray into Variable.

    If the input is Variable, it is returned as is.

    Params
    ------
    obj: ArrayLike
        Object to be converted into `Variable`

    Returns
    -------
    Variable(obj): Variable
        Variable object
    """
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")

        self.data = data
        self.name = name
        self.grad = None
        self.creator: Function | None = None
        self.generation = 0

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __neg__(self):
        return neg(self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return rsub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return rdiv(self, other)

    def __pow__(self, c):
        return pow(self, c)

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

    def set_creator(self, func: Function):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        """Propagate backwards to calculate the derivative.

        Params
        ------
        retain_grad: bool
            Whether to retain the gradadient.
        create_graph: bool
            Whether to create backward calculation graph.
            When you need to calculate the second or higher derivative, set this to True.
        """
        if self.grad is None:
            # Use Variable as grad instead of nodarray
            # so that calculation graph is made when backward() method is called
            # self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()

            # Make sure that f is not None.
            if f is None:
                raise ValueError("The function is None.")

            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]

            with using_config("enable_backdrop", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

                # When `retain_grad` is `False`, set the grads of outputs to `None`.
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self):
        return dezero.functions.transpose(self)

    @property
    def T(self):
        return dezero.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    def cleargrad(self):
        self.grad = None


class Function:
    def __call__(self, *inputs):
        # Transform ndarray inputs into `Variable`
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        # Notes: As you can see in the line below, the input args to
        # every `forward()` method in a Function subclass is ndarrays, not `Variable`s.
        # Therefore, we can use any operators and functions defined on ndarray
        # when we implement our own operators or functions by defining `forward()` methods.
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        # Record input/output of the function ONLY when `Config.enable_backdrop` is `True`
        if Config.enable_backdrop:
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        # If `outputs` consists of only 1 element, the content itself is returned,
        # otherwise whole tuple is returned.
        # return outputs if len(outputs) > 1 else outputs[0]
        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1)
    f = Add()
    return f(x0, x1)


class Neg(Function):
    def forward(self, x):
        y = -x
        return y

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1)
    f = Mul()
    return f(x0, x1)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        if x0.shape != x1.shape:
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        # x = self.inputs[0].data
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)
