"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
    with gzip.open(label_filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images.astype(np.float32) / 255.0, labels


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    # For each sample: loss = log(sum(exp(z_i))) - z_y
    # where z_y is the logit corresponding to the true label
    
    # Step 1: Compute exp(Z) for all logits
    exp_Z = ndl.exp(Z)
    
    # Step 2: Sum over classes (axis=1) to get sum(exp(z_i)) for each sample
    sum_exp_Z = ndl.summation(exp_Z, axes=(1,))
    
    # Step 3: Take log to get log(sum(exp(z_i))) for each sample
    log_sum_exp = ndl.log(sum_exp_Z)
    
    # Step 4: Extract z_y (the logit for the true label) for each sample
    # Multiply Z by y_one_hot (element-wise), then sum over classes
    # This gives us z_y for each sample
    z_y = ndl.summation(Z * y_one_hot, axes=(1,))
    
    # Step 5: Compute loss per sample: log(sum(exp(z_i))) - z_y
    loss_per_sample = log_sum_exp - z_y
    
    # Step 6: Average over all samples
    batch_size = Z.shape[0]
    avg_loss = ndl.summation(loss_per_sample, axes=None) / batch_size
    
    return avg_loss 


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_batches = (num_examples + batch - 1) // batch
    
    for i in range(num_batches):
        start_idx = i * batch
        end_idx = min((i + 1) * batch, num_examples)
        
        # Get batch data
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        
        # Convert to Tensor
        X_batch_tensor = ndl.Tensor(X_batch)
        
        # One-hot encode y_batch
        num_classes = W2.shape[1]
        y_one_hot = np.zeros((y_batch.shape[0], num_classes))
        y_one_hot[np.arange(y_batch.size), y_batch] = 1
        y_batch_tensor = ndl.Tensor(y_one_hot)
        
        # Forward pass: logits = ReLU(X * W1) * W2
        hidden = ndl.relu(ndl.matmul(X_batch_tensor, W1))
        logits = ndl.matmul(hidden, W2)
        
        # Compute loss
        loss = softmax_loss(logits, y_batch_tensor)
        
        # Backward pass
        loss.backward()
        
        # Update weights: W = W - lr * grad
        # Get gradients from W1 and W2
        W1_grad = W1.grad.numpy()
        W2_grad = W2.grad.numpy()
        
        # Update weights in numpy
        W1_numpy = W1.numpy() - lr * W1_grad
        W2_numpy = W2.numpy() - lr * W2_grad
        
        # Create new Tensors with updated weights
        W1 = ndl.Tensor(W1_numpy)
        W2 = ndl.Tensor(W2_numpy)
    
    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
