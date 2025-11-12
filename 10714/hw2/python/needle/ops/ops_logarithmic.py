from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        # LogSoftmax(z) = z - LogSumExp(z, axis=1)
        # Assuming 2D input and softmax over axis=1 (features dimension)
        max_z = array_api.max(Z, axis=1, keepdims=True)
        exp_z = array_api.exp(Z - max_z)
        sum_exp = array_api.sum(exp_z, axis=1, keepdims=True)
        log_sum_exp = array_api.log(sum_exp) + max_z
        return Z - log_sum_exp

    def gradient(self, out_grad: Tensor, node: Tensor):
        # Gradient: d/dz_j [logsoftmax(z)]_i = δ_ij - softmax(z)_j
        # For batched computation: grad = out_grad - softmax * sum(out_grad, axis=1, keepdims=True)
        z = node.inputs[0]
        # Compute softmax
        max_z = Tensor(z.realize_cached_data().max(axis=1, keepdims=True), device=z.device, dtype=z.dtype)
        exp_z = exp(z - max_z)
        sum_exp = summation(exp_z, axes=(1,)).reshape((z.shape[0], 1))
        softmax = exp_z / sum_exp.broadcast_to(exp_z.shape)
        # Compute gradient
        sum_out_grad = summation(out_grad, axes=(1,)).reshape((z.shape[0], 1))
        return out_grad - softmax * sum_out_grad.broadcast_to(out_grad.shape)


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        # Compute max along the specified axes, keeping dimensions
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        # Compute log(sum(exp(Z - max_z))) + max_z
        exp_z = array_api.exp(Z - max_z)
        sum_exp = array_api.sum(exp_z, axis=self.axes, keepdims=True)
        log_sum_exp = array_api.log(sum_exp) + max_z
        # Remove the keepdims dimension if axes were specified
        if self.axes is not None:
            return array_api.squeeze(log_sum_exp, axis=self.axes)
        else:
            return log_sum_exp.reshape(())

    def gradient(self, out_grad: Tensor, node: Tensor):
        # Gradient of logsumexp is softmax
        # d/dz_j [logsumexp(z)] = exp(z_j) / sum(exp(z_i)) = softmax(z)_j
        z = node.inputs[0]
        # Compute max for numerical stability
        max_z = Tensor(z.realize_cached_data().max(axis=self.axes, keepdims=True), device=z.device, dtype=z.dtype)
        # Compute exp(z - max_z)
        exp_z = exp(z - max_z)
        # Sum along axes
        sum_exp = summation(exp_z, axes=self.axes)
        # Broadcast sum_exp back to original shape for division
        # Need to reshape to add back the dimensions that were summed
        if self.axes is not None:
            shape = list(z.shape)
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            for axis in sorted(axes):
                shape[axis] = 1
            sum_exp = sum_exp.reshape(shape)
        # Compute softmax = exp_z / sum_exp
        softmax = exp_z / sum_exp.broadcast_to(exp_z.shape)
        # Gradient is out_grad * softmax
        # Need to broadcast out_grad to match softmax shape
        if self.axes is not None:
            shape = list(z.shape)
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            for axis in sorted(axes):
                shape[axis] = 1
            out_grad_reshaped = out_grad.reshape(shape)
            return out_grad_reshaped.broadcast_to(z.shape) * softmax
        else:
            return out_grad.broadcast_to(z.shape) * softmax


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)