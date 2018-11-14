import torch
import torch.nn as nn
import numpy as np
from layers import PoolLayer, DenseLayer
from collections import OrderedDict


batch_size = 2


""" Linear layer """
dim_in = 3
dim_out = 4

X = np.random.rand(batch_size, dim_in)
w = np.random.rand(dim_out, dim_in)
b = np.random.rand(dim_out)

lin_torch = nn.Linear(dim_in, dim_out, bias=True)
lin_torch.weight = nn.Parameter(torch.tensor(w))
lin_torch.bias = nn.Parameter(torch.tensor(b))
X_tensor = torch.tensor(X, requires_grad=True)
y_tensor = lin_torch(X_tensor)
loss_torch = y_tensor.sum()
loss_torch.backward()


lin = DenseLayer(dim_out, "gaussian")
lin.params["w"].value = w.T
lin.params["b"].value = b
X_dict = OrderedDict()
X_dict["data"] = X
y_dict = lin.forward(X_dict)
y_grad = OrderedDict()
y_grad["grad"] = np.ones(y_dict["data"].shape)
X_grad = lin.backward(y_grad)

print()
print("Pytorch linear layer output:")
print(y_tensor)
print()
print("Pytorch linear layer gradients:")
print(lin_torch.weight.grad / batch_size)
print(lin_torch.bias.grad / batch_size)
print(X_tensor.grad)
print()

print("Our linear layer output:")
print(y_dict["data"])
print()
print("Our linear layer gradients:")
print(lin.params["w"].grad)
print(lin.params["b"].grad)
print(X_grad["grad"])
print()


""" Pooling layer """
height = 5
width = 6
channels = 4
kernel_size = 3
stride = 1

X = np.random.rand(batch_size, height, width, channels)

pool_torch = nn.MaxPool2d(kernel_size, stride)
X_tensor = torch.tensor(X, requires_grad=True)
y_tensor = pool_torch(X_tensor.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
loss_torch = y_tensor[:, :3, :3, 0].sum()
loss_torch.backward()

pool = PoolLayer("maximum", kernel_size, stride, 0)
X_dict = OrderedDict()
X_dict["height"] = height
X_dict["width"] = width
X_dict["channels"] = channels
X_dict["data"] = X.reshape(batch_size, -1)
y_dict = pool.forward(X_dict)
y_grad = OrderedDict()
y_grad["grad"] = np.zeros(y_tensor.size())
y_grad["grad"][:, :3, :3, 0] = 1
y_grad["grad"] = y_grad["grad"].reshape(batch_size, -1)
X_grad = pool.backward(y_grad)

print("Pytorch pooling layer output:")
print(y_tensor.reshape(batch_size, -1))
print()
print("Pytorch pooling layer gradients:")
print(X_tensor.grad.reshape(batch_size, -1))
print()

print("Our pooling layer output:")
print(y_dict["data"])
print()
print("Our pooling layer gradients:")
print(X_grad["grad"])
print()