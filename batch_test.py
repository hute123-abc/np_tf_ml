import numpy as np
import torch
import torchvision

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=256, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=256, shuffle=True)


def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(y_pred, y, epsilon=1e-12):
    """
    y_pred is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
        Note that y is **not** one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    n = y.shape[0]
    p = softmax(y_pred)

    # avoid computing log(0)
    p = np.clip(p, epsilon, 1.)

    # We use multidimensional array indexing to extract
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[np.arange(n), y])
    loss = np.sum(log_likelihood) / n
    return loss


def grad_cross_entropy(y_pred, y):
    """
    y_pred is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    n = y.shape[0]
    grad = softmax(y_pred)

    grad[np.arange(n), y] -= 1
    grad = grad / n
    return grad


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 256, 784, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

n_epochs = 10
learning_rate = 1e-3
display_freq = 50

for t in range(n_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        # Forward pass: compute predicted y
        x = x.view(x.shape[0], -1)
        x, y = x.numpy(), y.numpy()
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # Compute and print loss
        loss = cross_entropy(y_pred, y)
        if batch_idx % display_freq == 0:
            print('epoch = {}\tbatch_idx = {}\tloss = {}'.format(t, batch_idx, loss))

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = grad_cross_entropy(y_pred, y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # Update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2