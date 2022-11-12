import torch 
import torchvision
import torch.nn as nn
import numpy as np 
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import os
from torch.autograd import Variable


# =======================================
#           Basic autograd example 1
# =======================================

# Create tensors.

x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph
y =  w*x + b   # y=2*x+3

# compute gradients
y.backward()

# print out the gradients.
print(x.grad)   #x.grad = 2
print(w.grad)   #w.grad = 1
print(b.grad)   #b.grad = 1


# =========================================
#           Basic autograd example 2
# =========================================

# create tensors of shape (10,3) and (10,2)
x = torch.randn(10,3)
y = torch.randn(10,2)

# build a fully connected layer.

linear = nn.Linear(3,2)
print('w:',linear.weight)
print('b:',linear.bias)

# Build loss function and optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(),lr=0.01)

# forward pass
pred = linear(x)

# compute loss
loss = criterion(pred,y)
print('Loss:',loss.item())

# Backward pass.
loss.backward()

# print out the gradients.
print('dL/dw :',linear.weight.grad)
print('dL/db :',linear.bias.grad)

# 1-step gradient descent.
optimizer.step()

# you can also perform gradient descent at low level
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# print out the loss after 1-step gradient descent
pred = linear(x)
loss =criterion(pred,y)
print('loss after 1- step optimization :',loss.item())


# ==================================================
#             Loading data from numpy
# ==================================================

# Create a numpy array
x = np.array([[1,2],[3,4]])

# convert numpy array to a torch tensor
y = torch.from_numpy(x)


# convert the torch tensor to a numpy array
z = y.numpy()

# =============================================================
#                 input pipeline
# =============================================================

# Download and construct CIFAR-10 dataset
train_dataset = dsets.CIFAR10(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

# Fetch one data pair (read data from disk).

image,label =train_dataset[0]
print(image.size())
print(label)


# Data loader (this provides queue and threads in a very simple way)
train_laoder = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=64,
                                    shuffle=True)
# When iterations starts , queue and thread start to load data from files.
data_iter = iter(train_laoder)

# Mini-Batch images and labels
images , labels = data_iter.next()

# Actual usage of the data loader is as below.
for images , labels in train_laoder:
    # Training code should be written here
    pass


# ==================================================================
#                     Input pipeline for custom Datasets
# ==================================================================


# you should build your custom datasets as below
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1.Initilize file paths or a list of the fila names
        pass
    def __getitem__(self,index):
        # TODO
        # 1.read one data from file (eg. using numpy.fromfile,PIL.Image.open)
        # 2.Preprocess the data (eg. torchvision.Transform)
        # 3.Return a data pair
        pass
    def __len__(self):
        # you should change 0 to the total size of your dataset.
        return 0

custom_dataset = CustomDataset()
train_laoder = torch.utils.data.DataLoader(dataset=custom_dataset,
                                            batch_size=64,
                                            shuffle=True)


# ==================================================================
#                     Pretrained Module
# ==================================================================


# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model ,set as below
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning
resnet.fc=nn.Linear(resnet.fc.in_features,100)   #100 is an example

#Forward pass.

images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)

print(outputs.size())    #(64, 100)

# =============================================================
#                     Save and load model
# ==============================================================

# save and load  the entire model
torch.save(resnet , 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended)
torch.save(resnet.state_dict(),'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))