"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.data + self.weight_decay * p.data
            if idx not in self.u:
                self.u[idx] = 0
            self.u[idx] = self.momentum * self.u[idx] + (1 - self.momentum) * grad
            p.data = p.data - self.lr * self.u[idx]

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        pass


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # Increment time step
        self.t += 1
        
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            # Get gradient with weight decay
            grad = p.grad.data + self.weight_decay * p.data
            
            # Initialize first moment (m) and second moment (v) if needed
            if idx not in self.m:
                self.m[idx] = 0
                self.v[idx] = 0
            
            # Update biased first moment estimate: m_t = β₁ * m_{t-1} + (1 - β₁) * grad
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate: v_t = β₂ * v_{t-1} + (1 - β₂) * grad²
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate: m̂_t = m_t / (1 - β₁^t)
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate: v̂_t = v_t / (1 - β₂^t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
            
            # Update parameters: θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
            p.data = p.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        ### END YOUR SOLUTION
