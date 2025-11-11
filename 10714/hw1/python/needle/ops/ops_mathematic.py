"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api
import needle as ndl

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        # For f(a, b) = a^b:
        # df/da = b * a^(b-1)
        # df/db = a^b * log(a)
        return out_grad * b * power(a, b - 1), out_grad * power(a, b) * log(a)

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (out_grad * self.scalar * a ** (self.scalar - 1),)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / rhs ** 2


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # Transpose reverses the order of two axes (axis1, axis2)
        # If axes=None, defaults to reversing the last two axes
        ndim = len(a.shape)
        
        # Determine which two axes to reverse
        if self.axes is None:
            # Default: reverse the last two axes
            if ndim < 2:
                return a  # No transpose needed for < 2 dimensions
            i, j = ndim - 2, ndim - 1
        elif len(self.axes) == 2:
            # Reverse the order of two specified axes
            i, j = self.axes
        else:
            raise ValueError(f"Transpose axes must be None or a tuple of 2 elements, got {self.axes}")
        
        # Build permutation: swap axes i and j
        numpy_axes = list(range(ndim))
        numpy_axes[i], numpy_axes[j] = numpy_axes[j], numpy_axes[i]
        return array_api.transpose(a, axes=tuple(numpy_axes))

    def gradient(self, out_grad, node):
        # To undo a transpose, we apply the same transpose operation
        # If forward was axes=(i,j), backward should also be axes=(i,j)
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        # For broadcast, we need to sum over the broadcast dimensions
        # The gradient should have the same shape as the original input
        original_shape = node.inputs[0].shape
        output_shape = out_grad.shape
        
        # Find dimensions that were broadcast (dimensions that were 1 in input but >1 in output)
        # Align shapes from right to left
        original_shape_list = list(original_shape)
        output_shape_list = list(output_shape)
        
        # Pad original_shape with 1s on the left if needed
        while len(original_shape_list) < len(output_shape_list):
            original_shape_list.insert(0, 1)
        
        # Find axes that were broadcast (1 -> >1)
        sum_axes = []
        for i in range(len(output_shape_list)):
            if original_shape_list[i] == 1 and output_shape_list[i] > 1:
                sum_axes.append(i)
        
        # Sum over broadcast dimensions
        if sum_axes:
            grad = summation(out_grad, axes=tuple(sum_axes))
        else:
            grad = out_grad
        
        # Reshape to match original input shape
        if grad.shape != original_shape:
            grad = reshape(grad, original_shape)
        
        return grad


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad, node):
        # For summation, we need to broadcast the gradient back to the original shape
        # If axes is None, sum all dimensions -> output is scalar, broadcast to original shape
        # If axes is specified, we need to reshape to insert dimensions at the summed axes
        original_shape = node.inputs[0].shape
        out_shape = out_grad.shape
        
        if self.axes is None:
            # Sum all dimensions -> output is scalar, broadcast to original shape
            return broadcast_to(out_grad, original_shape)
        else:
            # Build new shape by inserting 1 at the summed axes
            # Example: (5, 4) sum axes=(1,) -> (5,), need to reshape to (5, 1) then broadcast to (5, 4)
            new_shape = []
            out_idx = 0
            for i in range(len(original_shape)):
                if i in self.axes:
                    new_shape.append(1)
                else:
                    new_shape.append(out_shape[out_idx])
                    out_idx += 1
            # Reshape and then broadcast
            grad = reshape(out_grad, tuple(new_shape))
            return broadcast_to(grad, original_shape)


def summation(a, axes=None):
    return Summation(axes)(a)

class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        # For C = A @ B:
        # dA = dC @ B^T (with summation over broadcast dimensions if rhs has more dims)
        # dB = A^T @ dC (with summation over broadcast dimensions if lhs has more dims)
        
        grad_lhs = matmul(out_grad, transpose(rhs))
        grad_rhs = matmul(transpose(lhs), out_grad)
        
        # Sum over broadcast dimensions
        # If rhs has more dimensions than lhs, sum over the extra dimensions in grad_lhs
        if len(rhs.shape) > len(lhs.shape):
            broadcast_dims = tuple(range(len(rhs.shape) - len(lhs.shape)))
            grad_lhs = summation(grad_lhs, axes=broadcast_dims)
        
        # If lhs has more dimensions than rhs, sum over the extra dimensions in grad_rhs
        if len(lhs.shape) > len(rhs.shape):
            broadcast_dims = tuple(range(len(lhs.shape) - len(rhs.shape)))
            grad_rhs = summation(grad_rhs, axes=broadcast_dims)
        
        return grad_lhs, grad_rhs


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        # ReLU gradient: 1 if input > 0, else 0
        # Note: We can use realize_cached_data() here since ReLU is not twice differentiable
        input_data = node.inputs[0].realize_cached_data()
        mask = (input_data > 0).astype(input_data.dtype)  # 1 where input > 0, 0 otherwise
        return out_grad * ndl.Tensor(mask)


def relu(a):
    return ReLU()(a)

