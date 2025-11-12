"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, **kwargs))
        if bias:
            # Reshape before wrapping in Parameter to keep it as a Parameter
            bias_init = init.kaiming_uniform(out_features, 1, **kwargs).reshape((1, out_features))
            self.bias = Parameter(bias_init)
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        return X @ self.weight + (self.bias if self.bias is not None else 0).broadcast_to((X.shape[0], self.out_features))


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        return X.reshape((batch_size, -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # l_softmax(z, y) = logsumexp(z, axis=1) - z_y
        # logits shape: (batch_size, num_classes)
        # y shape: (batch_size,) containing class indices
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        
        # Create one-hot encoding of y
        y_one_hot = init.one_hot(num_classes, y, device=logits.device, dtype=logits.dtype)
        
        # Compute logsumexp over axis 1
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        
        # Select z_y using one-hot encoding: sum(logits * y_one_hot, axis=1)
        z_y = ops.summation(logits * y_one_hot, axes=(1,))
        
        # Compute loss: mean(logsumexp - z_y)
        loss = log_sum_exp - z_y
        return ops.summation(loss) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x shape: (batch_size, dim)
        batch_size = x.shape[0]
        
        if self.training:
            # Compute batch statistics: mean and variance over batch dimension (axis=0)
            mean = ops.summation(x, axes=(0,)).reshape((1, self.dim)) / batch_size
            mean_broadcast = mean.broadcast_to(x.shape)
            
            # Centered data
            x_centered = x - mean_broadcast
            
            # Variance: E[(x - mean)^2]
            variance = ops.summation(x_centered ** 2, axes=(0,)).reshape((1, self.dim)) / batch_size
            variance_broadcast = variance.broadcast_to(x.shape)
            
            # Update running statistics (using detached values to avoid building computation graph)
            self.running_mean = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data.reshape((self.dim,))
            self.running_var = (1 - self.momentum) * self.running_var.data + self.momentum * variance.data.reshape((self.dim,))
            
            # Normalize
            x_norm = x_centered / ops.power_scalar(variance_broadcast + self.eps, 0.5)
        else:
            # Use running statistics during evaluation
            mean_broadcast = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var_broadcast = self.running_var.reshape((1, self.dim)).broadcast_to(x.shape)
            x_norm = (x - mean_broadcast) / ops.power_scalar(var_broadcast + self.eps, 0.5)
        
        # Apply learnable parameters
        weight_broadcast = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias_broadcast = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        
        return weight_broadcast * x_norm + bias_broadcast
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x shape: (batch_size, dim)
        # Compute mean and variance over the feature dimension (axis=1)
        batch_size = x.shape[0]
        
        # Mean over features for each sample: shape (batch_size, 1)
        mean = ops.summation(x, axes=(1,)).reshape((batch_size, 1)) / self.dim
        mean_broadcast = mean.broadcast_to(x.shape)
        
        # Centered data
        x_centered = x - mean_broadcast
        
        # Variance: E[(x - mean)^2]
        variance = ops.summation(x_centered ** 2, axes=(1,)).reshape((batch_size, 1)) / self.dim
        variance_broadcast = variance.broadcast_to(x.shape)
        
        # Normalize
        x_norm = x_centered / ops.power_scalar(variance_broadcast + self.eps, 0.5)
        
        # Apply learnable parameters
        # Broadcast weight and bias from (dim,) to (batch_size, dim)
        weight_broadcast = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias_broadcast = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        
        return weight_broadcast * x_norm + bias_broadcast
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # Generate a random mask: 1 with probability (1-p), 0 with probability p
            mask = init.randb(*x.shape, p=1-self.p, dtype=x.dtype, device=x.device)
            # Scale by 1/(1-p) to maintain expected value
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
