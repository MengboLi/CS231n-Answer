# 这个脚本可以直接在装有cuda pytorch torchvision的机器上训练模型 
# 输入命令 “python 文件名”  就可以了
# 制定特定的GPU可以在python前加 这个命令 ：CUDA_VISIBLE_DEVICES=1  1可以换，从0开始，代表GPU编号

"""This is a py to run on  linux platform
	if you have a GPU,that will use GPU or use CPU.
	This is a modle train on cifar-10-batches-py after study cs231n class"""

'''t1 = time()
t2 = time()
print('Fast: %fs' % (t2 - t1)) '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F  # useful stateless functions

import numpy as np
from time import time

NUM_TRAIN = 49000 # train_sample num from train_set
dtype = torch.float32 # we will be using float throughout this tutorial
device = None
# To choose a device to train
def Device_Choose(Device = 'GPU'):
	global device
	if Device == 'GPU' and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print('using device:', device)

class Flatten(nn.Module):
    def forward(self, x):
        N = x.shape[0]
        return x.view(N, -1) # "flatten" the C * H * W values into a single vector per image
		
def check_accuracy_part(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train_part(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part(loader_val, model)
                print()

# Constant to control how frequently we print train loss		
print_every = 100
# LoadData
transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,
                             transform = transform_train)
loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
cifar10_val = dset.CIFAR10('./datasets', train=True, download=True,
                           transform = transform_test)
loader_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, 
                            transform = transform_test)
loader_test = DataLoader(cifar10_test, batch_size=64)

channel_1 = 64
channel_2 = 128
channel_3 = 256
hidden_layer_size = 3000
learning_rate = 10e-3

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=channel_1, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
    nn.Conv2d(in_channels=channel_1, out_channels=channel_1, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
    nn.MaxPool2d(kernel_size=2,stride = 2),
	
    nn.BatchNorm2d(channel_1),
    nn.Conv2d(in_channels=channel_1, out_channels=channel_2, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
    nn.Conv2d(in_channels=channel_2, out_channels=channel_2, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
    nn.MaxPool2d(kernel_size=2,stride = 2),
	
    nn.BatchNorm2d(channel_2),
    nn.Conv2d(in_channels=channel_2, out_channels=channel_3, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
    nn.Conv2d(in_channels=channel_3, out_channels=channel_3, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
    nn.MaxPool2d(kernel_size=2,stride = 2),
	
    nn.BatchNorm2d(channel_3),
    Flatten(),
    nn.Linear(channel_3*4*4,hidden_layer_size),
    nn.ReLU(),
    #nn.Dropout(p=0.2),
    nn.Linear(hidden_layer_size,10),
)
model.cuda()
# model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
# cudnn.benchmark = True
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True,weight_decay=5e-4)
# You should get at least 70% accuracy
Device_Choose('GPU')
t1 = time()
train_part(model, optimizer, epochs=20)
t2 = time()
print('The train time is : %fs' % (t2 - t1)) 

best_model = model
check_accuracy_part(loader_test, best_model)
torch.save(model, 'model.pkl')
print("Training complete,mode data has saved as 'model.pkl' in  current directory,have fun!")














